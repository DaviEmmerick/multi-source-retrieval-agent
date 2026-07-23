from typing import List, TypedDict, Optional
from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_TEMPERATURE_REWRITE, LLM_BASE_URL,
    FAISS_MODEL, FAISS_K, FAISS_SIMILARITY_THRESHOLD, MAX_REWRITE_ITERATIONS,
    MIN_DOC_LENGTH, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP
)
from logger import get_logger
import os

logger = get_logger(__name__)

# Iniciando conexão com o Neo4jgraph

graph = Neo4jGraph(
    url="neo4j://127.0.0.1:7687",
    username="neo4j",
    password="password123"
)


import numpy as np
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception as e:
    logger.warning(f"FAISS/Sentence-Transformers não disponível: {e}")
    faiss = None
    SentenceTransformer = None

_faiss_index: Optional[faiss.IndexFlatIP] = None
_faiss_embeddings: Optional[np.ndarray] = None
_faiss_docs: Optional[List[str]] = None
_sbert_model: Optional[SentenceTransformer] = None

class GraphState(TypedDict):
    question: str 
    documents: List[str] 
    iterations: int 
    generation: str 

# Transformando os textos em n-chunks
def process_pdf(file_path: str) -> List:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages) 
    return chunks

# Inicializando o SBERT
def init_faiss(model_name: str = None):
    global _sbert_model
    if model_name is None:
        model_name = FAISS_MODEL
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model

# Indexando os documentos no FAISS, normalizando as dimensões e transformando os chunks em embeddings
def index_documents(docs: List[str], model_name: str = None):
    global _faiss_index, _faiss_embeddings, _faiss_docs, _sbert_model
    if model_name is None:
        model_name = FAISS_MODEL
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")

    if _sbert_model is None:
        _sbert_model = init_faiss(model_name)

    texts = [t if isinstance(t, str) else str(t) for t in docs]
    embeddings = _sbert_model.encode(texts, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _faiss_index = index
    _faiss_embeddings = embeddings
    _faiss_docs = texts

    return len(texts)

# Busca semântica
def semantic_search(query: str, k: int = None):
    global _faiss_index, _sbert_model, _faiss_docs
    if k is None:
        k = FAISS_K
    if _faiss_index is None or _faiss_docs is None:
        return []
    if _sbert_model is None:
        _sbert_model = init_faiss()

    q_emb = _sbert_model.encode([query], convert_to_numpy=True) # Convertendo para arrays [223..55], [333..344]
    faiss.normalize_L2(q_emb) # Agora conseguimos usar similaridade do cos
    D, I = _faiss_index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(_faiss_docs):
            continue
        results.append(_faiss_docs[idx])
    return results


def retrieve(state: GraphState):
    logger.info(f"Iniciando recuperação para: {state['question']}")
    question = state["question"]
    
    try:
        llm = ChatOllama(
            model=LLM_MODEL, 
            temperature=LLM_TEMPERATURE,            
            baseUrl=LLM_BASE_URL
        )

        cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm, 
            graph=graph, 
            verbose=True,
            return_direct=True 
        )

        try:
            logger.debug(f"Executando Cypher query para: '{question}'...")
            graph_results = cypher_chain.invoke({"query": question})
            graph_context = f"Dados do Grafo: {str(graph_results)}"
            logger.info("Query Cypher executada com sucesso")
        except Exception as e:
            logger.warning(f"Erro na consulta Cypher: {e}. Tentando fallback...")
            graph_context = "Nenhum dado encontrado no grafo."

    except Exception as e:
        logger.error(f"Erro ao conectar ao Neo4j ou LLM: {e}")
        graph_context = "Nenhum dado encontrado (erro de conexão)."

    return {"documents": [graph_context]}


def hybrid_search(state: GraphState):

    question = state.get("question", "")
    logger.info("Iniciando busca híbrida (Neo4j + FAISS)")

    # 1) Recupera do grafo
    try:
        graph_docs = retrieve(state).get("documents", [])
        logger.debug(f"{len(graph_docs)} documentos recuperados do Neo4j")
    except Exception as e:
        logger.error(f"Erro ao recuperar do grafo: {e}")
        graph_docs = ["Nenhum dado encontrado."]

    # 2) Busca semântica no FAISS
    try:
        faiss_docs = semantic_search(question, k=FAISS_K)
        logger.debug(f"{len(faiss_docs)} documentos recuperados do FAISS")
    except Exception as e:
        logger.warning(f"Erro na busca semântica FAISS: {e}")
        faiss_docs = []

    combined = []
    combined.extend(graph_docs)
    combined.extend([d for d in faiss_docs if d not in graph_docs])
    
    logger.info(f"Busca híbrida concluída: {len(combined)} documentos consolidados")
    state["documents"] = combined
    return {"documents": combined}

def generate(state: GraphState):
    """Gera a resposta final usando o LLM com o contexto consolidado (grafo + FAISS).
    Faz tentativas seguras e reverte para placeholder caso o LLM falhe."""
    question = state.get("question", "")
    docs = state.get("documents", [])
    context = "\n\n".join(docs)

    prompt = f"Use o contexto abaixo para responder a pergunta de forma objetiva.\n\nContext:\n{context}\n\nQuestion: {question}\n\nResposta:" 

    generation = None
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            baseUrl=LLM_BASE_URL
        )
        logger.debug("Iniciando geração de resposta via LLM...")
        # Tentar diferentes formas de chamada para compatibilidade
        try:
            # Alguns bindings oferecem .generate()
            res = llm.generate([{"role": "user", "content": prompt}])
            generation = str(res)
        except Exception as e1:
            logger.debug(f"Método .generate() falhou, tentando .invoke(): {e1}")
            try:
                generation = llm(prompt)
            except Exception as e2:
                logger.warning(f"Ambos os métodos falharam: {e2}")
                generation = None
    except Exception as e:
        logger.error(f"Erro ao chamar LLM para gerar resposta: {e}")

    if not generation:
        logger.warning("Usando fallback para resposta (contexto apenas)")
        generation = f"[GENERATED ANSWER - fallback]\nQuestion: {question}\nContext excerpt:\n{context[:1500]}"
    else:
        logger.info("Resposta gerada com sucesso")

    state["generation"] = generation
    return {"generation": generation}


def rewrite(state: GraphState):
    """Reescreve/refina a query usando o LLM quando os documentos recuperados não forem relevantes."""
    question = state.get("question", "")
    iterations = state.get("iterations", 0) + 1
    
    # Limite máximo de iterações para evitar loops infinitos
    if iterations > MAX_REWRITE_ITERATIONS:
        logger.warning(f"Limite de reescrita atingido ({iterations} iterações). Usando query original.")
        return {"question": question, "iterations": iterations}

    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE_REWRITE,
            baseUrl=LLM_BASE_URL
        )
        
        prompt = f"""A pergunta original não retornou documentos relevantes:
        
Original: "{question}"

Gere uma pergunta alternativa que seja mais específica ou que tente abordagens diferentes de busca.
Responda APENAS com a pergunta reescrita, sem explicações adicionais."""

        rewritten = llm.invoke(prompt)
        rewritten_text = str(rewritten).strip()
        
        logger.info(f"🔄 Query reescrita (iteração {iterations}): {rewritten_text}")
        state["question"] = rewritten_text
        state["iterations"] = iterations
        return {"question": rewritten_text, "iterations": iterations}
        
    except Exception as e:
        logger.warning(f"Erro ao reescrever query com LLM: {e}. Usando heurística simples.")
        # Fallback para reescrita heurística
        rewritten = f"{question} (refinada, tentativa {iterations})"
        state["question"] = rewritten
        state["iterations"] = iterations
        return {"question": rewritten, "iterations": iterations}


def grade_documents(state: GraphState):
    """Avalia se os documentos recuperados são suficientes para responder.
    Usa similaridade semântica entre a pergunta e os documentos.
    Retorna o rótulo 'yes' ou 'no' para controlar o fluxo do StateGraph.
    """
    docs = state.get("documents", [])
    question = state.get("question", "")
    
    if not docs:
        logger.warning("Nenhum documento recuperado.")
        return "no"
    
    # Verifica se há documentos que claramente indicam erro
    error_docs = 0
    for d in docs:
        txt = (d or "").lower()
        if "nenhum dado encontrado" in txt or ("erro" in txt and "contexto" not in txt):
            error_docs += 1
    
    if error_docs == len(docs):
        logger.warning("Todos os documentos contêm erros ou estão vazios.")
        return "no"
    
    # Tenta usar similaridade semântica se SBERT está disponível
    try:
        if _sbert_model is not None and len(docs) > 0:
            q_emb = _sbert_model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            
            # Calcula similaridade com cada documento
            doc_embeddings = _sbert_model.encode(docs, convert_to_numpy=True)
            faiss.normalize_L2(doc_embeddings)
            
            # Similaridade média (Inner Product após normalização = cosine similarity)
            similarities = np.dot(doc_embeddings, q_emb.T).flatten()
            avg_similarity = float(np.mean(similarities))
            
            # Threshold: configurável
            threshold = FAISS_SIMILARITY_THRESHOLD
            logger.debug(f"Similaridade média: {avg_similarity:.3f} (threshold: {threshold})")
            
            if avg_similarity > threshold:
                logger.info(" Documentos são relevantes (similaridade > threshold)")
                return "yes"
            else:
                logger.info("Documentos não são relevantes o suficiente (similaridade baixa)")
                return "no"
    except Exception as e:
        logger.warning(f"Erro ao calcular similaridade semântica: {e}")
    
    # Fallback: verificação simples de conteúdo não-vazio
    for d in docs:
        txt = (d or "").lower()
        if len(txt.strip()) > MIN_DOC_LENGTH:
            logger.info(" Documentos contêm conteúdo significativo")
            return "yes"
    
    logger.info(" Nenhum documento com conteúdo significativo encontrado")
    return "no"


workflow = StateGraph(GraphState)

workflow.add_node("hybrid_search", hybrid_search)
workflow.add_node("generate_output", generate)
workflow.add_node("rewrite_query", rewrite)

workflow.set_entry_point("hybrid_search")

workflow.add_conditional_edges(
    "hybrid_search", 
    grade_documents, 
    {
        "yes": "generate_output", 
        "no": "rewrite_query"      
    }
)

workflow.add_edge("rewrite_query", "hybrid_search")
workflow.add_edge("generate_output", END)
app = workflow.compile()

if __name__ == "__main__":

    file_path = "finances.pdf" # teste
    meus_chunks = process_pdf(file_path)
    print(f"Total de fatias geradas: {len(meus_chunks)}")