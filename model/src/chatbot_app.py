"""
Streamlit Interface for NOVA Information Management School Chatbot

This module provides a web interface for the NOVA chatbot using Streamlit
with an improved chat-like UI.
"""

import streamlit as st
import sys
import os
from typing import List, Dict
import time

# Import your chatbot module
try:
    from chatbot import NOVAChatbot, create_chatbot
except ImportError:
    st.error("Could not import chatbot module. Make sure chatbot.py is in the same directory.")
    st.stop()


def initialize_chatbot():
    """Initialize the chatbot and cache it"""
    if 'chatbot' not in st.session_state:
        with st.spinner('🤖 Initializing NOVA Chatbot... This may take a few minutes.'):
            try:
                st.session_state.chatbot = create_chatbot()
                st.session_state.messages = []  # Store messages as list of dicts
                st.success('✅ Chatbot initialized successfully!')
                time.sleep(1)  # Brief pause to show success message
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {str(e)}")
                return False
    return True


def add_message(role: str, content: str, context: List[str] = None):
    """Add a message to the conversation"""
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time()
    }
    if context:
        message["context"] = context
    
    st.session_state.messages.append(message)


def display_message(message: Dict, is_latest: bool = False):
    """Display a single message with proper styling"""
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🎓"):
            if is_latest:
                # Use write_stream for typing effect on latest message
                st.write_stream(stream_text(message["content"]))
            else:
                st.write(message["content"])
            
            # Show context if available
            if message.get("context"):
                with st.expander("📚 View Source Context", expanded=False):
                    for i, context in enumerate(message["context"]):
                        st.markdown(f"**📄 Source {i+1}:**")
                        # Truncate long context for better readability
                        display_context = context[:400] + "..." if len(context) > 400 else context
                        st.markdown(f"```\n{display_context}\n```")
                        if i < len(message["context"]) - 1:
                            st.divider()


def stream_text(text: str):
    """Generator function to create typing effect"""
    words = text.split()
    for i in range(len(words)):
        yield " ".join(words[:i+1]) + " "
        time.sleep(0.02)  # Adjust speed as needed


def get_example_questions():
    """Return a list of example questions"""
    return [
        "What are the mandatory courses in the first year?",
        "Who teaches Data Mining?",
        "What elective courses are available in the second semester?",
        "Tell me about the Machine Learning course",
        "What courses does Professor Silva teach?",
        "What is the course structure for the Master's program?",
        "How many ECTS does each course have?",
        "What are the prerequisites for Advanced Analytics?"
    ]


def main():
    st.set_page_config(
        page_title="NOVA Chatbot",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    .example-button {
        margin: 0.2rem 0;
        width: 100%;
    }
    
    .stats-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e3f2fd;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 NOVA Information Management School Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize chatbot
    if not initialize_chatbot():
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        add_message("assistant", "👋 Hello! I'm the NOVA IMS Assistant. I'm here to help you with questions about courses, professors, and academic information. How can I assist you today?")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Chat Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            add_message("assistant", "👋 Hello! I'm the NOVA IMS Assistant. I'm here to help you with questions about courses, professors, and academic information. How can I assist you today?")
            st.rerun()
        
        # Chat statistics
        if len(st.session_state.messages) > 1:
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f"""
            <div class="stats-box">
                <h4>📊 Chat Stats</h4>
                <p>👤 Your questions: {user_messages}</p>
                <p>🤖 Bot responses: {len(st.session_state.messages) - user_messages}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Example questions
        st.header("💡 Example Questions")
        st.markdown("*Click on any question to try it:*")
        
        for i, example in enumerate(get_example_questions()):
            if st.button(f"💭 {example}", key=f"example_{i}", help="Click to ask this question"):
                # Add user message
                add_message("user", example)
                
                # Get bot response
                with st.spinner('🤔 Thinking...'):
                    try:
                        # Get chat history in the format the chatbot expects
                        chat_history = []
                        for msg in st.session_state.messages[:-1]:  # Exclude the just-added message
                            if msg["role"] == "user":
                                chat_history.append(f"User: {msg['content']}")
                            else:
                                chat_history.append(f"Assistant: {msg['content']}")
                        
                        result = st.session_state.chatbot.ask(example, chat_history)
                        
                        if "error" in result:
                            add_message("assistant", f"❌ Sorry, I encountered an error: {result['error']}")
                        else:
                            add_message("assistant", result['answer'], result.get('context'))
                        
                        st.rerun()
                    
                    except Exception as e:
                        add_message("assistant", f"❌ Sorry, I encountered an unexpected error: {str(e)}")
                        st.rerun()
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.info("""
        **This chatbot helps with:**
        
        📚 Course information & structure  
        👨‍🏫 Professor details  
        📅 Academic planning  
        📋 Requirements & prerequisites  
        🎯 Program guidance
        
        **💡 Tips:**
        - Be specific in your questions
        - Ask about courses, professors, or academic structure
        - Use clear, direct language
        """)
    
    # Main chat area
    st.header("💬 Chat")
    
    # Display chat messages (excluding the one being processed)
    for message in st.session_state.messages:
        display_message(message, is_latest=False)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about NOVA IMS courses and professors..."):
        # Add user message
        add_message("user", prompt)
        
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner('🤔 Thinking...'):
                try:
                    # Get chat history in the format the chatbot expects
                    chat_history = []
                    for msg in st.session_state.messages[:-1]:  # Exclude the just-added user message
                        if msg["role"] == "user":
                            chat_history.append(f"User: {msg['content']}")
                        else:
                            chat_history.append(f"Assistant: {msg['content']}")
                    
                    result = st.session_state.chatbot.ask(prompt, chat_history)
                    
                    if "error" in result:
                        response = f"❌ Sorry, I encountered an error: {result['error']}"
                        st.write(response)
                        add_message("assistant", response)
                    else:
                        # Stream the response with typing effect
                        full_response = ""
                        response_placeholder = st.empty()
                        
                        for chunk in stream_text(result['answer']):
                            full_response = chunk
                            response_placeholder.write(full_response)
                        
                        # Add context expander if available
                        context_placeholder = st.empty()
                        if result.get('context'):
                            with context_placeholder.container():
                                with st.expander("📚 View Source Context", expanded=False):
                                    for i, context in enumerate(result['context']):
                                        st.markdown(f"**📄 Source {i+1}:**")
                                        display_context = context[:400] + "..." if len(context) > 400 else context
                                        st.markdown(f"```\n{display_context}\n```")
                                        if i < len(result['context']) - 1:
                                            st.divider()
                        
                        # Add message to session state after streaming is complete
                        add_message("assistant", result['answer'], result.get('context'))
                
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an unexpected error: {str(e)}"
                    st.write(error_msg)
                    add_message("assistant", error_msg)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "🎓 <em>Developed for NOVA Information Management School</em> | "
        "Powered by AI 🤖"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()