from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from setup_neo4j import graph
import os

class GraphState(TypedDict):
    question: str 
    documents: List[str] 
    iterations: int 
    generation: str 

def process_pdf(file_path: str) -> List:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages) 
    return chunks


def retrieve(state: GraphState):
    print("Iniciando a etapa de recuperação de documentos relevantes para a pergunta:", state["question"])
    question = state["question"]
    
    llm = ChatOllama(
        model="qwen2.5-coder:3b", 
        temperature=0.1,            
        baseUrl="http://localhost:11434"
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True,
        return_direct=True 
    )

    try:
        print(f"Qwen gerando query Cypher para: '{question}'...")
        graph_results = cypher_chain.invoke({"query": question})
        graph_context = f"Dados do Grafo: {str(graph_results)}"
    except Exception as e:
        print(f"Erro na consulta com Qwen: {e}")
        graph_context = "Nenhum dado encontrado."

    return {"documents": [graph_context]}

def generate(state: GraphState):
    pass

def rewrite(state: GraphState):
    pass


def grade_documents(state: GraphState):
    pass


workflow = StateGraph(GraphState)

workflow.add_node("hybrid_search", retrieve)
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

    file_path = "finances.pdf"
    meus_chunks = process_pdf(file_path)
    print(f"Total de fatias geradas: {len(meus_chunks)}")