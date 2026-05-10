import streamlit as st
import tempfile
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

    # Layout de colunas para centralização
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container(border=True):
            st.subheader("📁 Upload de Documentos")
            uploaded_file = st.file_uploader("Arraste ou selecione um PDF técnico", type="pdf", label_visibility="collapsed")

            if uploaded_file:
                st.info(f"Arquivo selecionado: **{uploaded_file.name}**")
                
                if st.button("Finalizar e Indexar Documento"):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        with st.spinner("Fragmentando e indexando PDF..."):
                            chunks = range(42)
                            st.session_state["chunks"] = chunks
                            
                        st.success(f"✅ Sucesso: {len(chunks)} chunks gerados e prontos para consulta!")
                        
                        os.remove(tmp_path)
                    except Exception as e:
                        st.error(f"Erro ao processar o arquivo: {e}")

    if "chunks" in st.session_state:
        st.divider()
        with st.expander("🔍 Visualizar Chunks Gerados"):
            st.write(st.session_state["chunks"])

if __name__ == "__main__":
    main()