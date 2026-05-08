
import streamlit as st
import os
import time
from pyvis.network import Network
import streamlit.components.v1 as components
from src.assistant import CodeAssistant
from src.graph_store import CodeNode
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="GAC-RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
    }
    .stButton>button:hover {
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .chat-bubble {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    .user-bubble {
        background-color: #1e2a3a;
    }
    .assistant-bubble {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_nodes" not in st.session_state:
    st.session_state.last_nodes = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 GAC-RAG")
    st.markdown("---")
    
    repo_path = st.text_input("Repository Path", value="./sample_repo")
    llm_provider = st.selectbox("LLM Provider", ["anthropic", "openai", "google"])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize"):
            with st.spinner("Initializing GAC-RAG..."):
                st.session_state.assistant = CodeAssistant(
                    repo_path=repo_path, 
                    llm_provider=llm_provider
                )
                st.success("Assistant ready!")
                
    with col2:
        if st.button("Re-Index"):
            if st.session_state.assistant:
                with st.spinner("Indexing repository..."):
                    st.session_state.assistant.index(clear_existing=True)
                    st.success("Index complete!")
            else:
                st.error("Init first!")

    st.markdown("---")
    st.subheader("Settings")
    max_hops = st.slider("Max Hops", 1, 4, 2)
    top_k = st.slider("Top-K Anchors", 1, 10, 5)
    
    if st.session_state.assistant:
        st.session_state.assistant.retriever.max_hops = max_hops
        st.session_state.assistant.retriever.top_k_anchor = top_k

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🕸️ Graph Explorer", "📁 Node Inspector"])

with tab1:
    st.title("Code Intelligence Chat")
    
    # Chat display
    for message in st.session_state.chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        css_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
        st.markdown(f"""
            <div class="chat-bubble {css_class}">
                <strong>{role}:</strong><br>{message['content']}
            </div>
        """, unsafe_allow_html=True)

    # Input area
    query = st.chat_input("Ask a question about your codebase...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Use the streaming API we built earlier
            if st.session_state.assistant:
                # We need to capture the retrieved nodes for the other tabs
                candidates = st.session_state.assistant.retriever.retrieve(query)
                final_nodes, _ = st.session_state.assistant.reranker.rerank(query, candidates)
                st.session_state.last_nodes = final_nodes
                
                # Manual streaming display for Streamlit
                context = st.session_state.assistant.retriever.format_context(final_nodes)
                for chunk in st.session_state.assistant.stream_ask(query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                st.rerun()
            else:
                st.error("Please initialize the assistant in the sidebar.")

with tab2:
    st.title("Retrieval Graph")
    if st.session_state.last_nodes:
        st.write(f"Visualizing relationship between {len(st.session_state.last_nodes)} retrieved nodes.")
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        
        nodes = st.session_state.last_nodes
        node_ids = {n.id for n in nodes}
        
        for n in nodes:
            color = "#4a90e2" if n.kind == "function" else "#f5a623" if n.kind == "class" else "#7ed321"
            net.add_node(n.id, label=n.name, title=f"File: {n.file_path}\nKind: {n.kind}", color=color, size=25)
            
        # Add edges (simplified check for demo)
        for n in nodes:
            # In a real app, we'd query Neo4j for actual edges between these specific nodes
            # Here we'll just show some placeholder edges or if the graph store supports it
            pass
            
        net.repulsion(node_distance=200, spring_length=200)
        
        # Save and display
        path = "graph.html"
        net.save_graph(path)
        HtmlFile = open(path, 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=650)
    else:
        st.info("Ask a question in the Chat tab to see the retrieved context graph.")

with tab3:
    st.title("Node Inspector")
    if st.session_state.last_nodes:
        for node in st.session_state.last_nodes:
            with st.expander(f"[{node.kind.upper()}] {node.name} (Score: {node.score:.3f})"):
                st.code(node.source_code, language=node.language)
                st.markdown(f"**File:** `{node.file_path}`")
                st.markdown(f"**Lines:** `{node.start_line} - {node.end_line}`")
                if node.docstring:
                    st.info(node.docstring)
    else:
        st.info("No nodes retrieved yet.")

# --- Footer ---
st.markdown("---")
st.markdown("GAC-RAG: Graph-Augmented Code Retrieval-Augmented Generation")
