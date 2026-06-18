app_code = """
# TODO: Build your Streamlit chatbot application

import streamlit as st

# your code here
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Zyro Dynamics HR Bot", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk")
st.markdown("Ask any question regarding company policies, leave, or conduct.")

# --- RAG Setup (Cached to prevent reloading on every interaction) ---
@st.cache_resource
def setup_rag():
    # Path to documents
    path = "/kaggle/input/zyro-dynamics-hr-corpus/"
    
    # Load and Chunk
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    # Initialize LLM (Ensure secrets are set in your deployment environment)
    # Note: Replace with your chosen provider logic from Cell 9
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.1) 
    
    return retriever, llm

try:
    retriever, llm = setup_rag()
except Exception as e:
    st.error("Please ensure your API keys are configured and the dataset is attached.")
    st.stop()

# --- Helpers ---
def get_response(question):
    # Guardrail
    oos_prompt = ChatPromptTemplate.from_template("Is this HR-related? Answer 'IN' or 'OUT': {question}")
    guard_chain = oos_prompt | llm | StrOutputParser()
    if "OUT" in guard_chain.invoke({"question": question}).upper():
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []

    # RAG Chain
    docs = retriever.invoke(question)
    context = "\\n\\n".join([d.page_content for d in docs])
    rag_prompt = ChatPromptTemplate.from_template("Context: {context}\\n\\nQuestion: {question}\\nAnswer:")
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How many sick leaves can I take?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response, sources = get_response(prompt)
        st.markdown(response)
        
        if sources:
            with st.expander("View Sources"):
                for doc in sources:
                    st.write(f"- {doc.metadata.get('source', 'Policy Document')}")

    st.session_state.messages.append({"role": "assistant", "content": response})
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
