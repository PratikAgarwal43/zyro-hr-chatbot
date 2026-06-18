import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbedembeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Zyro Dynamics HR Bot", page_icon="📑")
st.title("Zyro Dynamics HR Help Desk")
st.markdown("Ask any question regarding company policies, leave, or conduct.")

# --- 1. Secure API Key Loading ---
# This looks for GROQ_API_KEY inside your Streamlit Cloud App Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please configure your GROQ_API_KEY in the Streamlit Secrets setting.")
    st.stop()

# --- 2. Initialize Vector Database ---
@st.cache_resource
def initialize_vector_db():
    # Since your PDFs are in the root directory next to app.py, use "./"
    loader = PyPDFDirectoryLoader("./")
    docs = loader.load()
    
    if not docs:
        st.error("No PDF documents found in the repository!")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(final_documents, embeddings)
    return vector_store

with st.spinner("Processing HR Documents..."):
    vectors = initialize_vector_db()

# --- 3. Chat Interface ---
if vectors:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_question := st.chat_input("How can I help you today?"):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Set up LLM Chain
        llm = ChatGroq(model_name="llama3-8b-8192")
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert HR assistant for Zyro Dynamics. Answer the question based strictly on the provided context. 
        If you do not know the answer, politely state that you cannot find it in the company policies.
        
        Context:
        {context}
        
        Question: {input}
        """)
        
        # Retrieve context from FAISS
        retriever = vectors.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.invoke(user_question)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        chain = prompt | llm | StrOutputParser()
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke({"context": context_text, "input": user_question})
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
