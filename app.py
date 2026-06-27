import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk")
st.markdown("Ask any question regarding company compliance, leaves, WFH arrangements, or corporate conduct.")

# --- Cached RAG Pipeline Synchronization Initialization ---
@st.cache_resource
def setup_rag():
    # Points to the local execution path root folder where the repo files are deployed
    path = "./" 
    
    # Document Parsing and Processing Flow
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    
    if not documents:
        raise FileNotFoundError("No PDF compliance materials found in the target repository root path.")
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Index Generation and Storage
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Production-Tier Context Retrieval Window Framework matching notebook configs
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5}
    )
    
    # High-Parameter Flagship Model Execution
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    
    return retriever, llm

try:
    retriever, llm = setup_rag()
except Exception as e:
    st.error(f"Initialization Configuration Error: {str(e)}")
    st.info("Ensure your GROQ_API_KEY is configured correctly in your Streamlit Cloud Advanced Secrets environment console.")
    st.stop()

# --- Execution Core Helper Layer ---
def get_response(question):
    # Guardrail Check Protocol Configuration
    oos_prompt = ChatPromptTemplate.from_template(
        "Analyze the user query. Determine if it is explicitly related to corporate employment, workforce structures, "
        "benefits, or conduct rules. Cross-policy queries or multi-rule comparisons are strictly IN_SCOPE. "
        "Respond with ONLY 'IN_SCOPE' or 'OUT_OF_SCOPE': {question}\nAnswer:"
    )
    guard_chain = oos_prompt | llm | StrOutputParser()
    if "OUT_OF_SCOPE" in guard_chain.invoke({"question": question}).upper():
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []

    # Dynamic Retrieval Assembly Frame
    docs = retriever.invoke(question)
    context = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])
    
    # Authoritative Compliance-Grade Operational Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
    You are a precise HR Compliance Officer for Zyro Dynamics Pvt. Ltd.
    Answer the employee's question with absolute factual accuracy based ONLY on the internal policy context below. Do not extrapolate or introduce structural pleasantries.
    
    Context:
    {context}
    
    Question: {question}
    Answer:""")
    
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

# --- Interactive Chat UI Flow ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question regarding Zyro Dynamics compliance rules..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response, sources = get_response(prompt)
        st.markdown(response)
        
        if sources:
            with st.expander("View Document Citations"):
                for doc in sources:
                    filename = os.path.basename(doc.metadata.get('source', 'Policy Document'))
                    st.write(f"- {filename}")

    st.session_state.messages.append({"role": "assistant", "content": response})