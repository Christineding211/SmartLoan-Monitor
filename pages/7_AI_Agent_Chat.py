"""AI Governance Agent - Chat Interface"""
import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Get the directory containing this file (pages/)
current_file = os.path.abspath(__file__)
pages_dir = os.path.dirname(current_file)
project_root = os.path.dirname(pages_dir)  # github files/

# Add project root to path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env from project root
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Now import should work
try:
    from smartloan_agent.agent_core import GovernanceAgent
    from smartloan_agent.agent_tools import get_current_metrics
except ImportError as e:
    st.error(f"Import failed: {e}")
    st.error(f"Project root: {project_root}")
    st.error(f"Python path: {sys.path[:3]}")
    st.stop()

# Page config
st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Title 
st.title("🤖 AI Governance Agent")
st.markdown("""
Chat with your ML monitoring agent. Ask about drift, performance, fairness, or compliance.
*Powered by OpenAI GPT-4o mini*
""")

# Initialize agent - CHANGE THIS
@st.cache_resource
def get_agent():
    try:
        # Will automatically load from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("⚠️ OPENAI_API_KEY not found in .env file")
            st.info("Add to your .env file: `OPENAI_API_KEY='sk-...'`")
            return None
        
        return GovernanceAgent(api_key=api_key)
        
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

agent = get_agent()

if agent is None:
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about model health, drift, fairness, or compliance..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):
            response = agent.chat(prompt)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with quick actions
st.sidebar.title("⚡ Quick Actions")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🏥 Health Check", use_container_width=True):
        with st.spinner("Running health check..."):
            response = agent.run_daily_check()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

with col2:
    if st.button("📊 Weekly Audit", use_container_width=True):
        with st.spinner("Running weekly audit..."):
            response = agent.run_weekly_audit()
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if st.sidebar.button("🔄 Clear Chat", use_container_width=True):
    st.session_state.messages = []
    agent.clear_history()
    st.rerun()

# Example queries
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Example Queries")

examples = [
    "What's the current model performance?",
    "Are there any drifted features?",
    "Run a fairness audit for income_group",
    "Generate a compliance report",
    "What issues need immediate attention?",
    "Explain the PSI scores in simple terms",
    "Is the model ready for the audit?"
]

for example in examples:
    if st.sidebar.button(f"💬 {example}", key=example, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example})
        with st.spinner("🤔 Analyzing..."):
            response = agent.chat(example)
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

# Show current metrics summary
try:
    metrics = get_current_metrics.invoke({})
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
        st.metric("PSI Max", f"{metrics.get('psi_max_value', 0):.3f}")
    
    with col2:
        st.metric("KS", f"{metrics.get('ks', 0):.3f}")
        fairness_status = "✅" if metrics.get('pass_80_rule') else "⚠️"
        st.metric("Fairness", fairness_status)
        
except Exception as e:
    st.sidebar.info("Run a batch to see metrics")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
**Agent Capabilities:**
- Real-time monitoring
- Drift detection
- Fairness auditing
- Compliance reporting
- Natural language queries
</small>
""", unsafe_allow_html=True)