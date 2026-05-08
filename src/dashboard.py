
import streamlit as st
import os
import time
import json
from pyvis.network import Network
import streamlit.components.v1 as components
from src.assistant import CodeAssistant
from src.graph_store import CodeNode
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="GAC-RAG | Enterprise Code Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium "Enterprise Classic" CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.875rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Custom Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
        border: none;
        color: white;
    }

    /* Chat Styling */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .user-bubble {
        background-color: #eff6ff;
        border: 1px solid #dbeafe;
        color: #1e40af;
        margin-left: 2rem;
    }

    .assistant-bubble {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #334155;
        margin-right: 2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .node-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if "assistant" not in st.session_state:
    st.session_state.assistant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_nodes" not in st.session_state:
    st.session_state.last_nodes = []
if "insights" not in st.session_state:
    st.session_state.insights = None

# --- Header ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🏛️ GAC-RAG")
    st.caption("Graph-Augmented Code Retrieval & Multi-Agent Synthesis")
with col_h2:
    if st.session_state.assistant:
        st.success("● Engine Online")
    else:
        st.warning("○ Engine Offline")

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ System Configuration")
    repo_path = st.text_input("Project Root", value="./sample_repo")
    provider = st.selectbox("Intelligence Provider", ["google", "openai", "anthropic"])
    
    st.markdown("---")
    if st.button("🚀 Initialize System", use_container_width=True):
        try:
            with st.spinner("Configuring neural graph..."):
                st.session_state.assistant = CodeAssistant(repo_path=repo_path, llm_provider=provider)
                st.session_state.insights = st.session_state.assistant.get_codebase_insights()
                st.success("System ready!")
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")

    if st.session_state.assistant:
        if st.button("⚡ Re-Index Repository", use_container_width=True):
            with st.spinner("Analyzing codebase structure..."):
                st.session_state.assistant.index(clear_existing=True)
                st.session_state.insights = st.session_state.assistant.get_codebase_insights()
                st.success("Indexing complete!")

    st.markdown("---")
    st.subheader("🔍 Search Parameters")
    max_hops = st.slider("Graph Traversal Hops", 1, 5, 2)
    top_k = st.slider("Vector Candidates", 1, 15, 5)
    
    if st.session_state.assistant:
        st.session_state.assistant.retriever.max_hops = max_hops
        st.session_state.assistant.retriever.top_k_anchor = top_k

# --- Dashboard Metrics ---
if st.session_state.insights:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Nodes</div><div class="metric-value">{st.session_state.insights['total_nodes']}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Relationships</div><div class="metric-value">{st.session_state.insights['total_edges']}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Graph Density</div><div class="metric-value">{st.session_state.insights['health'].get('avg_degree', 0):.2f}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Orphan Files</div><div class="metric-value">{st.session_state.insights['health'].get('orphan_nodes', 0)}</div></div>""", unsafe_allow_html=True)

# --- Main Content ---
tabs = st.tabs(["💬 Intelligence Chat", "🕸️ Knowledge Graph", "🧬 Codebase Insights", "📁 Explorer"])

with tabs[0]:
    # Chat Interface
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    if query := st.chat_input("Ask about architecture, logic, or dependencies..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        if st.session_state.assistant:
            with st.chat_message("assistant"):
                # Retrieval
                candidates = st.session_state.assistant.retriever.retrieve(query)
                final_nodes, _ = st.session_state.assistant.reranker.rerank(query, candidates)
                st.session_state.last_nodes = final_nodes
                
                # Streaming Answer
                response_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.assistant.stream_ask(query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                st.rerun()
        else:
            st.error("Please initialize the system in the sidebar.")

with tabs[1]:
    st.subheader("Structural Dependency Graph")
    if st.session_state.last_nodes:
        net = Network(height="600px", width="100%", bgcolor="#f8fafc", font_color="#0f172a", directed=True)
        for n in st.session_state.last_nodes:
            color = "#3b82f6" if n.kind == "function" else "#f59e0b" if n.kind == "class" else "#10b981"
            net.add_node(n.id, label=n.name, title=f"File: {n.file_path}", color=color, size=20)
        
        net.toggle_physics(True)
        path = "graph_enterprise.html"
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)
    else:
        st.info("Retrieve context via chat to visualize the sub-graph.")

with tabs[2]:
    st.subheader("Architectural Hubs")
    if st.session_state.insights:
        hubs = st.session_state.insights.get('top_hubs', [])
        if hubs:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Top Connected Components (High Centrality)")
                df_hubs = [{"Name": h['name'], "Kind": h['kind'], "Connections": h['degree']} for h in hubs]
                st.table(df_hubs)
            with col_b:
                # Simple distribution
                kind_data = st.session_state.insights['health'].get('by_kind', {})
                st.write("Component Distribution")
                st.bar_chart(kind_data)
        else:
            st.info("No hubs detected. Try re-indexing.")
    else:
        st.info("Initialize system to see codebase insights.")

with tabs[3]:
    st.subheader("Code Inspector")
    if st.session_state.last_nodes:
        for node in st.session_state.last_nodes:
            with st.expander(f"{node.kind.upper()}: {node.name}"):
                st.text(f"Path: {node.file_path}")
                st.code(node.source_code, language=node.language)
                if node.docstring:
                    st.markdown(f"**AI Summary:** *{node.docstring}*")
    else:
        st.info("Nodes will appear here after a query.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>GAC-RAG Enterprise v2.1 | Powered by Gemini 1.5 Pro & Neo4j</div>", unsafe_allow_html=True)
