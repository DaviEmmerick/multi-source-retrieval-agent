from typing import List, TypedDict
from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
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
    pass

def generate(state: GraphState):
    pass

def rewrite(state: GraphState):
    pass


def grade_documents(state: GraphState):
    pass


workflow = StateGraph(GraphState)

workflow.add_node("vector_search", retrieve)
workflow.add_node("generate_output", generate)
workflow.add_node("rewrite_query", rewrite)

workflow.set_entry_point("vector_search")

workflow.add_conditional_edges(
    "vector_search", 
    grade_documents, 
    {
        "yes": "generate_output", 
        "no": "rewrite_query"      
    }
)

workflow.add_edge("rewrite_query", "vector_search")
workflow.add_edge("generate_output", END)
app = workflow.compile()

if __name__ == "__main__":

    file_path = "finances.pdf"
    meus_chunks = process_pdf(file_path)
    print(f"Total de fatias geradas: {len(meus_chunks)}")