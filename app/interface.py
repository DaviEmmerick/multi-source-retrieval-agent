import streamlit as st
import tempfile
import os
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer

from graph import process_pdf, index_documents, init_faiss, app as workflow_app, hybrid_search, graph

from config import STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT, LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL
from logger import get_logger
import os
logger = get_logger(__name__)

def main():
    # Configuração da página (Recomendado layout='wide' para apps com sidebar)
    st.set_page_config(page_title=STREAMLIT_PAGE_TITLE, page_icon="🧠", layout="wide")

    # Inicialização do estado da sessão
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [] # Histórico de chat

    # CSS Customizado para refinamento visual
    st.markdown("""
        <style>
        .stChatMessage {
            border-radius: 10px;
            padding: 15px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Estilização suave para containers de aviso/info */
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # BARRA LATERAL (SIDEBAR) - GESTÃO DE DADOS
    # ==========================================
    with st.sidebar:
        st.title("⚙️ Gestão de Conhecimento")
        st.markdown("Faça o upload da base de conhecimento técnica para alimentar o RAG.")
        
        st.divider()
        
        st.subheader("📁 Upload de PDF")
        uploaded_file = st.file_uploader("Selecione um documento técnico", type="pdf", label_visibility="collapsed")

        if uploaded_file:
            st.markdown("<br>", unsafe_allow_html=True) # Adiciona um respiro visual no lugar do aviso verde
            
            if st.button("🚀 Processar e Indexar Documento", use_container_width=True, type="primary"):
                
                # Interface de Status para feedback visual passo a passo
                with st.status("Processando documento...", expanded=True) as status:
                    try:
                        # 1. Cria o arquivo temporário
                        st.write("📥 Lendo arquivo...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        # 2. Gera as fatias reais de texto
                        st.write("✂️ Fragmentando PDF...")
                        chunks = process_pdf(tmp_path)
                        st.session_state["chunks"] = [c.page_content for c in chunks]
                        logger.info(f"PDF processado: {len(chunks)} chunks gerados")

                        # 3. Indexa no FAISS local
                        st.write("🗂️ Indexando vetores no FAISS...")
                        try:
                            init_faiss()
                            indexed = index_documents(st.session_state["chunks"])
                            logger.info(f"{indexed} chunks indexados no FAISS local")
                        except Exception as e:
                            logger.warning(f"Falha ao indexar em FAISS: {e}")
                            st.warning(f"Aviso FAISS: {e}")

                        # 4. Processamento de Grafo
                        st.write("🕸️ Extraindo entidades e gerando Grafo (Neo4j)...")
                        llm = ChatOllama(
                            model=LLM_MODEL,
                            temperature=LLM_TEMPERATURE,
                            baseUrl=LLM_BASE_URL
                        )
                        transformer = LLMGraphTransformer(llm=llm)
                        graph_documents = transformer.convert_to_graph_documents(chunks)
                        
                        # 5. Salva no Neo4j
                        st.write("💾 Injetando dados no Neo4j...")
                        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=False)
                        
                        # Limpeza
                        os.remove(tmp_path)
                        
                        status.update(label=f"Sucesso! {len(chunks)} chunks processados.", state="complete", expanded=False)
                        st.toast('Base de conhecimento atualizada com sucesso!', icon='✅')

                    except Exception as e:
                        logger.error(f"Erro ao processar: {e}")
                        status.update(label="Erro no processamento", state="error", expanded=True)
                        st.error(f"Detalhes do erro: {e}")
        
        # Novo design para as métricas da base na sidebar
        if st.session_state.get("chunks"):
            st.divider()
            st.caption("📊 Status da Base de Conhecimento")
            st.metric(label="Fragmentos Indexados", value=len(st.session_state["chunks"]))
            with st.expander("🔍 Inspecionar Texto"):
                for idx, text in enumerate(st.session_state["chunks"]):
                    st.caption(f"**#{idx + 1}**: {text[:100]}...")

    # ==========================================
    # ÁREA PRINCIPAL - INTERFACE DE CHAT RAG
    # ==========================================
    st.markdown("<h1 style='text-align: center;'>🧠 Next-Gen Agentic RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Assistente inteligente alimentado por Grafos de Conhecimento e Busca Vetorial</p>", unsafe_allow_html=True)
    st.write("---")

    # Renderiza o histórico de mensagens sem os documentos de contexto
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrada de texto do usuário tipo Chat
    if user_question := st.chat_input("Faça uma pergunta sobre os documentos indexados..."):
        
        # 1. Adiciona e exibe a pergunta do usuário
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # 2. Processa a resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando grafos e buscando vetores..."):
                try:
                    logger.info(f"Processando pergunta: {user_question}")
                    
                    # Chamada ao fluxo (Workflow LangGraph/LangChain)
                    result = workflow_app.invoke({
                        "question": user_question,
                        "documents": [],
                        "iterations": 0,
                        "generation": ""
                    })

                    # Tratamento de fallback caso resultado seja None
                    if not result:
                        result = {"generation": "Erro: Não foi possível obter uma resposta do modelo.", "documents": [], "iterations": 0}

                    answer = result.get("generation", "Sem resposta gerada.")
                    docs = result.get("documents", [])
                    iterations = result.get("iterations", 0)

                    # Exibe a resposta principal
                    st.markdown(answer)

                    # Badge de status se houve reescrita
                    if iterations > 0:
                        st.caption(f"🔄 *A query foi otimizada {iterations} vez(es) durante a busca.*")

                    # 3. Salva no histórico para manter a conversa visível
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": answer,
                        "docs": docs
                    })

                except Exception as e:
                    error_msg = f"Ocorreu um erro ao processar sua pergunta: `{str(e)}`"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()