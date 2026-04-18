"""
chatbot_ui.py
=============
This component renders the MedBot chatbot interface at the bottom of the
medicine information page.

Features:
  - Message history displayed as styled chat bubbles (bot on left, user on right)
  - 8 pre-defined quick question chips for common queries (dosage, side effects, etc.)
  - Free-text input box + Send button
  - Intent badge on each bot response showing what the chatbot understood
  - Export button to download the full conversation as a text file
  - Medical disclaimer card

How it connects to the AI:
  When the user sends a message, handle_user_input() calls get_chat_response()
  from modules/chatbot.py. The chatbot preprocesses the text, classifies the
  intent using TF-IDF cosine similarity, and returns a response filled with
  the current medicine's data.

Session state:
  Chat history is stored in st.session_state.chat_history so it persists
  across Streamlit reruns. It is cleared when a new medicine is selected.

Author:  Mohammad Fayas Khan
Course:  INT428 — AI Systems Design
Version: 1.0.0
"""


import streamlit as st
from modules.chatbot import get_chat_response
from modules.export_utils import chat_history_to_text, create_download_button

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# Pre-defined quick question buttons shown below the chat input
QUICK_QUESTIONS = [
    "What is this medicine used for?",
    "What are the side effects?",
    "What is the dosage?",
    "Is it safe during pregnancy?",
    "Can children take this?",
    "What are the drug interactions?",
    "What are the alternatives?",
    "How should I store this medicine?",
]

# Maximum chat history messages before auto-trimming oldest
MAX_CHAT_HISTORY = 50


# ─────────────────────────────────────────────────────────────────────
# MESSAGE HANDLING
# ─────────────────────────────────────────────────────────────────────

def handle_user_input(user_text: str, medicine: dict, vectorizer, matrix, intent_labels: list) -> None:
    st.session_state.chat_history.append({"role": "user", "text": user_text, "intent": ""})
    st.session_state.pending_bot_response = True
    st.session_state.last_user_query = user_text
    st.rerun()

# ─────────────────────────────────────────────────────────────────────
# QUICK QUESTION CHIPS
# ─────────────────────────────────────────────────────────────────────

def render_quick_chips(medicine: dict, vectorizer, matrix, intent_labels: list) -> None:
    st.markdown(
        '<div style="font-family:var(--font-mono);font-size:0.68rem;'
        'color:var(--muted);letter-spacing:0.1em;margin-bottom:0.5rem;text-align:center;">'
        '⚡ QUICK QUESTIONS'
        '</div>',
        unsafe_allow_html=True
    )
    num_cols = 4
    for i in range(0, len(QUICK_QUESTIONS), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(QUICK_QUESTIONS):
                question = QUICK_QUESTIONS[idx]
                with col:
                    if st.button(question, key=f"chip_{idx}", use_container_width=True, help=f"Ask: {question}"):
                        handle_user_input(question, medicine, vectorizer, matrix, intent_labels)

# ─────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────

def render_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer-card" style="margin-top:2rem;">'
        '<div class="disclaimer-text" style="text-align:center;">'
        '⚠️ <strong>Medical Disclaimer:</strong> This application provides general '
        'medicine information only and is <strong>not a substitute for professional '
        'medical advice</strong>.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────
# MAIN CHATBOT RENDERER
# ─────────────────────────────────────────────────────────────────────

def render_chatbot(medicine: dict, vectorizer, matrix, intent_labels: list) -> None:
    med_name = str(medicine.get("name", "this medicine")).title()
    
    from modules.chatbot import get_chat_response, stream_response
    
    st.markdown(
        '<div class="section-header" style="margin-top:2rem;text-align:center;border-bottom:none;">💬 MEDBOT ASSISTANT</div>',
        unsafe_allow_html=True
    )
    
    if "pending_bot_response" not in st.session_state:
        st.session_state.pending_bot_response = False

    # Container for messages
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    f"👋 Hello! I'm **MedBot**, your offline assistant.\n\n"
                    f"I'm loaded with information about **{med_name}**. "
                    f"Ask me anything!"
                )
        else:
            for msg in st.session_state.chat_history:
                avatar = "🤖" if msg["role"] == "bot" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["text"])
                    if msg.get("intent") and msg["intent"] not in ["unknown", ""]:
                        st.caption(f"📌 {msg['intent'].replace('_', ' ').title()}")
        
        # Handle pending bot response stream
        if st.session_state.get("pending_bot_response"):
            user_text = st.session_state.last_user_query
            response_data = get_chat_response(user_text, medicine, vectorizer, matrix, intent_labels)
            
            with st.chat_message("assistant", avatar="🤖"):
                # Stream it out
                stream = stream_response(response_data["response"])
                text = st.write_stream(stream)
                if response_data["intent"] not in ["unknown", ""]:
                    st.caption(f"📌 {response_data['intent'].replace('_', ' ').title()}")
                
            # Append to history
            st.session_state.chat_history.append({
                "role": "bot",
                "text": response_data["response"],
                "intent": response_data["intent"],
                "confidence": response_data["confidence"]
            })
            
            # Reset state
            st.session_state.pending_bot_response = False
            st.rerun()

    # Chat Input 
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if prompt := st.chat_input(f"Ask about {med_name}..."):
        handle_user_input(prompt, medicine, vectorizer, matrix, intent_labels)

    # Quick chips below the chat box
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_quick_chips(medicine, vectorizer, matrix, intent_labels)

    if st.session_state.chat_history:
        transcript = chat_history_to_text(st.session_state.chat_history, medicine_name=med_name)
        create_download_button(
            data=transcript,
            filename=f"medscan_chat_{med_name.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            label="⬇ Export Chat Transcript"
        )
        
    render_disclaimer()
