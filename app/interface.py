import streamlit as st
import tempfile
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from graph import process_pdf 
from setup_neo4j import graph  
import os

def main():
    st.set_page_config(page_title="Agentic RAG", layout="wide")

    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #007bff;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Next-Gen Agentic RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Processe seus documentos técnicos com inteligência</p>", unsafe_allow_html=True)

    st.write("##")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container(border=True):
            st.subheader("📁 Upload de Documentos")
            uploaded_file = st.file_uploader("Arraste ou selecione um PDF técnico", type="pdf", label_visibility="collapsed")

            if uploaded_file:
                st.info(f"Arquivo selecionado: **{uploaded_file.name}**")
                
                if st.button("Finalizar e Indexar Documento"):
                    try:
                        # 1. Cria o arquivo temporário para o loader ler
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        with st.spinner("Fragmentando PDF e extraindo links de grafo via Qwen..."):
                            # 2. Gera as fatias reais de texto usando sua função importada
                            chunks = process_pdf(tmp_path)
                            
                            st.session_state["chunks"] = [c.page_content for c in chunks]
                            
                            llm = ChatOllama(
                                model="qwen2.5-coder:3b",
                                temperature=0.1,
                                baseUrl="http://localhost:11434"
                            )

                            transformer = LLMGraphTransformer(llm=llm)

                            graph_documents = transformer.convert_to_graph_documents(chunks)
                                        
                            # 6. Salva as bolinhas e setas geradas direto no Neo4j
                            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=False)
                            
                        st.success(f"✅ Sucesso: {len(chunks)} chunks gerados e Grafo injetado no Neo4j!")
                        
                        # Limpa o arquivo temporário do sistema operacional
                        os.remove(tmp_path)
                    except Exception as e:
                        st.error(f"Erro ao processar o arquivo: {e}")

    if "chunks" in st.session_state:
        st.divider()
        with st.expander("🔍 Visualizar Chunks Gerados (Texto Puro)"):
            for idx, text in enumerate(st.session_state["chunks"]):
                st.markdown(f"**Chunk #{idx + 1}**")
                st.code(text, language="text")

if __name__ == "__main__":
    main()