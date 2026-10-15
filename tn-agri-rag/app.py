import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(
    page_title="TN Agriculture Scheme Assistant",
    page_icon="🌾",
    layout="centered"
)

# Load Vector Store and set up modern RAG Chain (cached so it loads only once)
@st.cache_resource
def load_rag_pipeline():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load your local ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
    )
    
    # Strict prompt template ensuring it only uses local DB context
    system_prompt = (
        "You are an expert AI agricultural advisor helping farmers in Tamil Nadu understand government schemes, subsidies, and application guidelines.\n"
        "Answer the farmer's query **strictly and exclusively** using the context provided below.\n"
        "Do not use any external knowledge or web search. If the answer is not present in the context, politely reply:\n"
        "\"I'm sorry, but I can only provide information available in the official scheme documents loaded into my database.\"\n\n"
        "Context from official documents:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Modern LangChain combination and retrieval chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# UI Layout
st.title("🌾 Tamil Nadu Agriculture Scheme Assistant")
st.markdown("Ask questions about Tamil Nadu agricultural subsidies, welfare schemes, eligibility, and application guidelines.")

try:
    rag_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Error loading Vector Store. Please make sure you ran `python ingest.py` first! Details: {e}")
    st.stop()

# User input query form
query = st.text_input("Enter your query (e.g., *What are the requirements for micro-irrigation?*):", "")

if st.button("Search Schemes") and query:
    with st.spinner("Searching official scheme documents..."):
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        source_docs = response["context"]
        
        # Display the Answer
        st.markdown("### Answer:")
        st.success(answer)
        
        # Display Source Transparency
        with st.expander("View Source Documents & Pages"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page')}):**")
                st.text(doc.page_content[:300] + "...") # Show preview of chunk