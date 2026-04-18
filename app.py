
import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SemanticDoubleMergingSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
os.environ["GOOGLE_API_KEY"]="AQ.A" #your api key
# --- UI CONFIGURATION (Cinematic Dark Mode) ---
st.set_page_config(page_title="Enterprise AI Brain", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    .stTextInput > div > div > input { background-color: #161b22; color: #00ff41; border: 1px solid #30363d; }
    h1 { color: #00ff41; font-family: 'Courier New', Courier, monospace; }
    .stChatMessage { border-radius: 10px; border: 1px solid #30363d; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📂 Advanced Enterprise RAG")
st.subheader("Multi-Source Knowledge Retrieval System")


# --- NEW: FILE UPLOAD SECTION ---
if not os.path.exists("./data"):
    os.makedirs("./data")

uploaded_files = st.file_uploader("Upload Documents to Knowledge Base", 
                                  type=['pdf', 'txt', 'docx'], 
                                  accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Successfully saved {len(uploaded_files)} files. Now click 'Re-Index' in the sidebar!")
# --- INITIALIZE LLM & EMBEDDINGS ---
# Ensure your OPENAI_API_KEY is set in your PowerShell environment variables
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# --- CORE RAG LOGIC ---
def initialize_index():
    # 1. Create a data directory if it doesn't exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/welcome.txt", "w") as f:
            f.write("Welcome to your Enterprise RAG system. Upload documents to the data folder to begin.")

    # 2. Setup Persistent Vector Storage (ChromaDB)
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("enterprise_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Load Documents from local 'data' folder
    documents = SimpleDirectoryReader("./data").load_data()

    # 4. Create/Load Index
    if not os.path.exists("./chroma_db"):
        # Create new index if DB doesn't exist
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context,
            show_progress=True
        )
    else:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
    return index

# Initialize Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ LLMOps Controls")
    st.info("Status: Connected to ChromaDB")
    if st.button("Re-Index Documents"):
        with st.spinner("Processing local data..."):
            st.session_state.index = initialize_index()
            st.success("Index Updated!")

# --- MAIN CHAT INTERFACE ---
if "index" not in st.session_state:
    st.session_state.index = initialize_index()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your enterprise data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query_engine = st.session_state.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        response = query_engine.query(prompt)
        st.markdown(response.response)
        
        # Display Citations/Sources for GitHub 'Wow' factor
        with st.expander("🔍 View Source Context"):
            for node in response.source_nodes:
                st.write(f"**Source:** {node.metadata.get('file_name', 'Unknown')}")
                st.caption(node.get_content()[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": response.response})
