"""
app.py — Streamlit UI for the Personal RAG Chat Agent
"""

import os
import streamlit as st
from rag_agent import build_graph, PERSONA_NAME

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"Chat with {PERSONA_NAME}",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 780px; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f2027 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #63b3ed;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* Chat messages */
.chat-message {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-message.user { flex-direction: row-reverse; }
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar.bot  { background: #ffffff; border: 1px solid rgba(200,200,200,0.4); ; }
.avatar.user { background: linear-gradient(135deg, #1a3a2a, #0d2b1a); border: 1px solid rgba(72,187,120,0.3); }
.bubble {
    max-width: 82%;
    padding: 0.85rem 1.1rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.6;
}
.bubble.bot {
    background: #1e2640;
    border: 1px solid rgba(99,179,237,0.15);
    color: #cbd5e0;
    border-top-left-radius: 4px;
}
.bubble.user {
    background: #1a3a2a;
    border: 1px solid rgba(72,187,120,0.2);
    color: #c6f6d5;
    border-top-right-radius: 4px;
    text-align: right;
}

/* Suggestion chips */
.chips-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 1.2rem;
}
.chip {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.25);
    color: #90cdf4;
    font-size: 0.8rem;
    padding: 5px 14px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover {
    background: rgba(99,179,237,0.18);
    border-color: rgba(99,179,237,0.5);
}

/* Input area */
.stTextInput input {
    background: #1a2035 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    color: #e2e8f0 !important;
    caret-color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextInput input:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.1) !important;
}
.stTextInput input::placeholder {
    color: rgba(255, 255, 255, 0.35) !important;
    -webkit-text-fill-color: rgba(255, 255, 255, 0.35) !important;
    font-style: italic;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #1a4a8a) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(43,108,176,0.4) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(99,179,237,0.1) !important;
}
section[data-testid="stSidebar"] * { color: #a0aec0 !important; }
section[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; font-family: 'Syne', sans-serif !important; }

/* Spinner */
.stSpinner > div { border-top-color: #63b3ed !important; }

/* Dark background */
.stApp { background: #0a0e1a; }

/* Divider */
hr { border-color: rgba(99,179,237,0.1) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "myprofile.txt")


SUGGESTED_QUESTIONS = [
    "What are your top skills?",
    "Tell me about your work experience",
    "What projects have you built?",
    "What are your hobbies?",
    "What is your email?",
    "What is your education?",
    "Tell me about your AI projects",
    "Any Certifications?",
]


@st.cache_resource(show_spinner=False)
def get_agent():
    return build_graph(PROFILE_PATH)


def render_message(role: str, content: str):
    emoji = "🤖" if role == "assistant" else "🙋"
    css_role = "bot" if role == "assistant" else "user"
    st.markdown(
        f"""
    <div class="chat-message {css_role}">
        <div class="avatar {css_role}">{emoji}</div>
        <div class="bubble {css_role}">{content}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown(
    f"""
<div class="hero-header">
    <p class="hero-title">👋 Hi, I'm {PERSONA_NAME}</p>
    <p class="hero-subtitle">Ask me anything about my background, skills, or projects.</p>
    
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="
    background: rgba(255,193,7,0.08);
    border: 1px solid rgba(255,193,7,0.3);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    margin-bottom: 1.2rem;
    font-size: 0.83rem;
    color: #f6e05e;
    line-height: 1.6;
">
    ⚠️ <strong>Disclaimer:</strong> I have been working on new projects and upskilling continuously — some information retrieved here may not reflect my latest experience. 
    I'd love to <strong>chat with you directly</strong> to discuss my skills and achievements in detail! Model used might hallucinate on general info in some cases. 
</div>
""",
    unsafe_allow_html=True,
)

# Init state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Hey there! 👋 I'm an AI assistant that can help you with questions related to **{PERSONA_NAME}**. Ask me about her experience, skills, projects, or anything else you'd like to know!",
        }
    )

# Render history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# Suggestion chips (show only when no real conversation yet)

with st.sidebar:
    st.markdown("### 💡 Suggested Questions")
    st.markdown("---")
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if st.button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# Clickable buttons for suggestions


st.markdown("---")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder=f"Ask me anything about {PERSONA_NAME}...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    with col3:
        clear = st.form_submit_button("🗑️", use_container_width=True)

# After the form, handle clear:
if clear:
    st.session_state.messages = []
    st.rerun()

# Handle pending question from chip click
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
    submitted = True

# Process message
if submitted and user_input and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message("user", user_input)

    # Run agent
    with st.spinner("Thinking..."):
        try:
            agent = get_agent()
            result = agent.invoke(
                {
                    "question": user_input,
                    "context": [],
                    "answer": "",
                    "chat_history": st.session_state.messages[
                        :-1
                    ],  # history without latest
                }
            )
            answer = result["answer"]
        except Exception as e:
            answer = f"⚠️ Sorry, I ran into an error: `{str(e)}`\n\nCheck that your LLM token is valid and has Inference API access."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    render_message("assistant", answer)
    st.rerun()
