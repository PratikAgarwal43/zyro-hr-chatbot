app_code = """
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Zyro Dynamics HR Bot", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk")
st.markdown("Ask any question regarding company policies, leave, or conduct.")

# --- RAG Setup (Cached to prevent reloading on every interaction) ---
@st.cache_resource
def setup_rag():
    # Points to your local GitHub repository path root directory
    path = "./" 
    
    # Load and Chunk
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    
    if not documents:
        raise FileNotFoundError("No PDF compliance files were found in the repository root directory.")
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Matching notebook optimization settings (k=8)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.5}
    )
    
    # Initializing flagship open-source reasoning standard
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
    
    return retriever, llm

try:
    retriever, llm = setup_rag()
except Exception as e:
    st.error(f"Configuration Error: {str(e)}")
    st.info("Ensure your GROQ_API_KEY is configured in the Streamlit Advanced Secrets panel.")
    st.stop()

# --- Helpers ---
def get_response(question):
    # Guardrail Check
    oos_prompt = ChatPromptTemplate.from_template(
        "Analyze the user query. Respond with 'IN_SCOPE' if it is regarding company policies, leave, conduct, WFH, data security or benefits. "
        "Otherwise respond with 'OUT_OF_SCOPE'. Query: {question}\\nClassification:"
    )
    guard_chain = oos_prompt | llm | StrOutputParser()
    if "OUT_OF_SCOPE" in guard_chain.invoke({"question": question}).upper():
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []

    # Context Generation
    docs = retriever.invoke(question)
    context = "\\n\\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\\nContent: {d.page_content}" for d in docs])
    
    # Compliance Prompting Model
    rag_prompt = ChatPromptTemplate.from_template(\"\"\"
    You are a precise HR Compliance Officer for Zyro Dynamics Pvt. Ltd.
    Answer the employee's question with absolute factual accuracy based ONLY on the internal policy context below. Do not assume or extrapolate.
    
    Context:
    {context}
    
    Question: {question}
    Answer:\"\"\")
    
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

# --- Chat Interface ---
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
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
