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

def handle_user_input(user_text: str, medicine: dict,
                      vectorizer, matrix, intent_labels: list) -> None:
    """
    Process a user message, get a bot response, and update chat history.

    Steps:
      1. Validate input is not empty
      2. Append user message to chat_history
      3. Get chatbot response via get_chat_response()
      4. Append bot response to chat_history
      5. Auto-trim if history exceeds MAX_CHAT_HISTORY
      6. Trigger Streamlit rerun to refresh display

    Args:
        user_text (str): Raw user question string.
        medicine (dict): Currently displayed medicine data dict.
        vectorizer: Fitted TF-IDF vectoriser from build_intent_index.
        matrix: TF-IDF pattern matrix.
        intent_labels (list[str]): Intent label list.
    """
    user_text = user_text.strip()
    if not user_text:
        return

    # Step 1: Append user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "text": user_text,
        "intent": "",
    })

    # Step 2: Get chatbot response from NLP engine
    response_data = get_chat_response(
        user_input=user_text,
        medicine=medicine,
        vectorizer=vectorizer,
        matrix=matrix,
        intent_labels=intent_labels,
    )

    # Step 3: Append bot response with intent metadata
    st.session_state.chat_history.append({
        "role": "bot",
        "text": response_data["response"],
        "intent": response_data["intent"],
        "confidence": response_data["confidence"],
    })

    # Step 4: Auto-trim oldest messages if history is too long
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        # Keep only the most recent MAX_CHAT_HISTORY messages
        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

    # Step 5: Force Streamlit rerender with updated history
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# MESSAGE BUBBLE RENDERING
# ─────────────────────────────────────────────────────────────────────

def render_message_bubble(role: str, text: str, intent: str = "") -> None:
    """
    Render a single chat message bubble with correct alignment and styling.

    Bot messages: left-aligned, dark card2 background, cyan border.
    User messages: right-aligned, deep teal gradient.

    Args:
        role (str): "bot" or "user".
        text (str): Message content (may contain Markdown).
        intent (str): Detected intent tag (bot only, for badge display).
    """
    if role == "bot":
        sender_label = "🤖 MedBot"
        bubble_class = "msg-bot"
    else:
        sender_label = "👤 You"
        bubble_class = "msg-user"

    # Intent badge for bot messages (only when intent is known)
    intent_badge = ""
    if role == "bot" and intent and intent != "unknown":
        intent_display = intent.replace("_", " ").title()
        intent_badge = f'<div class="intent-badge">📌 {intent_display}</div>'

    # Render message bubble HTML
    st.markdown(
        f'<div class="{bubble_class}">'
        f'<div class="msg-sender">{sender_label}</div>'
        f'<div class="msg-text">{text}</div>'
        f'{intent_badge}'
        f'</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────
# QUICK QUESTION CHIPS
# ─────────────────────────────────────────────────────────────────────

def render_quick_chips(medicine: dict, vectorizer,
                       matrix, intent_labels: list) -> None:
    """
    Render quick question chips as clickable Streamlit buttons.

    Each chip sends a pre-defined question to the chatbot when clicked.
    Displayed in two rows of 4 columns for a chip-flow layout.

    Args:
        medicine (dict): Currently displayed medicine.
        vectorizer: TF-IDF vectoriser.
        matrix: TF-IDF matrix.
        intent_labels (list[str]): Intent labels.
    """
    st.markdown(
        '<div style="font-family:var(--font-mono);font-size:0.68rem;'
        'color:var(--muted);letter-spacing:0.1em;margin-bottom:0.5rem;">'
        '⚡ QUICK QUESTIONS'
        '</div>',
        unsafe_allow_html=True
    )

    # Arrange chips in rows of 4
    num_cols = 4
    for i in range(0, len(QUICK_QUESTIONS), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(QUICK_QUESTIONS):
                question = QUICK_QUESTIONS[idx]
                with col:
                    # Unique key per button to avoid Streamlit duplicate-key errors
                    if st.button(
                        question,
                        key=f"chip_{idx}",
                        use_container_width=True,
                        help=f"Ask: {question}"
                    ):
                        handle_user_input(question, medicine, vectorizer, matrix, intent_labels)


# ─────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────

def render_disclaimer() -> None:
    """
    Render the medical disclaimer below the chatbot interface.

    Displayed in an amber-tinted card to draw attention.
    """
    st.markdown(
        '<div class="disclaimer-card">'
        '<div class="disclaimer-text">'
        '⚠️ <strong>Medical Disclaimer:</strong> This application provides general '
        'medicine information only and is <strong>not a substitute for professional '
        'medical advice</strong>. Always consult a qualified healthcare professional '
        '(doctor, pharmacist, or specialist) before starting, stopping, or changing '
        'any medication. In case of emergency, dial 108 (India) or your local emergency number.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN CHATBOT RENDERER
# ─────────────────────────────────────────────────────────────────────

def render_chatbot(medicine: dict, vectorizer, matrix, intent_labels: list) -> None:
    """
    Render the complete MedBot chatbot interface.

    Structure:
      1. Section header
      2. Chat title bar with medicine name
      3. Scrollable message history container
      4. Quick question chips
      5. User text input + Send button
      6. Export chat button
      7. Medical disclaimer

    Args:
        medicine (dict): Currently displayed medicine data dict.
        vectorizer: Fitted TF-IDF vectoriser from build_intent_index.
        matrix: TF-IDF pattern matrix.
        intent_labels (list[str]): Intent label list.
    """
    med_name = medicine.get("name", "this medicine").title()

    # ── Section Divider ───────────────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="margin-top:2rem;">■ MEDSCAN MEDBOT — AI CHAT ASSISTANT</div>',
        unsafe_allow_html=True
    )

    # ── Chat Title Bar ────────────────────────────────────────────────
    st.markdown(
        f'<div class="chat-wrap">'
        f'<div class="chat-title">'
        f'<span style="font-size:1.4rem;">🤖</span>'
        f'<div>'
        f'<div class="chat-title-text">MedBot</div>'
        f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);">'
        f'Ask anything about {med_name}'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Message History ───────────────────────────────────────────────
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

    # Welcome message if history is empty
    if not st.session_state.chat_history:
        render_message_bubble(
            "bot",
            f"👋 Hello! I'm **MedBot**, your offline medicine assistant.\n\n"
            f"I'm loaded with information about **{med_name}**. "
            f"Ask me anything — side effects, dosage, interactions, pregnancy safety, and more!\n\n"
            f"*All responses are generated locally — no internet required.*",
            intent=""
        )
    else:
        # Render all messages in history
        for msg in st.session_state.chat_history:
            render_message_bubble(
                role=msg.get("role", "bot"),
                text=msg.get("text", ""),
                intent=msg.get("intent", "")
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Chips ───────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    render_quick_chips(medicine, vectorizer, matrix, intent_labels)

    # ── Text Input + Send ─────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_input, col_send = st.columns([5, 1])

    with col_input:
        user_input = st.text_input(
            label="Ask MedBot",
            placeholder=f"Ask about {med_name}... (e.g. 'What are side effects?')",
            key="chat_input",
            label_visibility="collapsed"
        )

    with col_send:
        send_clicked = st.button("Send ▶", key="chat_send", use_container_width=True)

    # Process input when Send is clicked or Enter pressed (text not empty)
    if send_clicked and user_input:
        handle_user_input(user_input, medicine, vectorizer, matrix, intent_labels)

    # ── Export Chat Button ────────────────────────────────────────────
    if st.session_state.chat_history:
        transcript = chat_history_to_text(
            st.session_state.chat_history,
            medicine_name=med_name
        )
        create_download_button(
            data=transcript,
            filename=f"medscan_chat_{med_name.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            label="⬇ Export Chat Transcript"
        )

    # ── Medical Disclaimer ────────────────────────────────────────────
    render_disclaimer()
