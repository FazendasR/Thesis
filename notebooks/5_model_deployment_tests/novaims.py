import os
# Move to Thesis directory (two levels up)
os.chdir(os.path.abspath(os.path.join("..", "..")))

# Move to model/src if it exists
model_dir = os.path.join(os.getcwd(), "model", "src")
if os.path.exists(model_dir):
    os.chdir(model_dir)

print("Current Directory:", os.getcwd())

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from libs import data_handeling as dh
from libs.settings import data_catalog as dc
from libs import data_retrievers as dr

# Load documents at module level
documents_chunked_with_ids_and_metadata = dh.load_documents_from_pickle(dc.DOCUMENTS_CHUNKED_WITH_IDS_AND_METADATA)

def move_id_to_metadata(documents):
    for doc in documents:
        if hasattr(doc, 'id') and doc.id is not None:
            doc.metadata['id'] = doc.id
    return documents

documents_chunked_with_ids_and_metadata = move_id_to_metadata(documents_chunked_with_ids_and_metadata)

# Page configuration
st.set_page_config(
    page_title="NOVA IMS AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stSpinner > div > div {
        border-color: #1f77b4 transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# Define the State class
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[str]

@st.cache_resource
def initialize_model():
    """Initialize the model and tokenizer (cached for performance)"""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    with st.spinner("🔄 Loading AI model... This may take a few minutes on first run."):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=3000,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = HuggingFacePipeline(
            pipeline=text_generation_pipeline,
            model_id=model_id,
            callbacks=callback_manager,
        )
    
    return llm

@st.cache_resource
def initialize_retrievers(_documents_chunked_with_ids_and_metadata):
    """Initialize the retrieval system"""
    with st.spinner("🔍 Loading document retrieval system..."):
        try:
            # Load your retrievers exactly as in your original code
            tfidf_retriever = dr.load_sparse_retriever(
                retriever_type="TF-IDF", 
                documents_chunked=_documents_chunked_with_ids_and_metadata, 
                top_k=5
            )
            chroma_retriever = dr.load_vector_retriever(
                collection_name="parent_documents_with_ids_and_metadata_embedded_v2", 
                top_k=5
            )
            hybrid_retriever = dr.load_hybrid_retriever(
                tfidf_retriever, 
                chroma_retriever, 
                weight_sparse=0.5, 
                weight_vector=0.5
            )
            hybrid_retriever_reranking = dr.get_reranking(hybrid_retriever, top_n=2)
            return hybrid_retriever_reranking
        except Exception as e:
            st.error(f"Error initializing retrievers: {str(e)}")
            st.stop()

def create_prompt_template():
    """Create the prompt template"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Role Description:
    You are a friendly and knowledgeable AI assistant developed by NOVA Information Management School. Your purpose is to support students by answering their questions accurately and effectively. 

    Guiding Principles:
    When the question is about the course structure of a semester or year:
    - If exists, count and list the mandatory courses name and organize them in bullet format.
    - If exists, count and list the elective courses name and organize them in bullet format.
    - IF a course name appears before both 'mandatory' and 'elective', consider it in both lists.
    - Suggest the user visit the official course page for full details.
    - If the question refers to a year, include courses of both semesters of that year. If no semester is mentioned, include both semesters and clearly label them.

    CRITICAL INSTRUCTIONS:
    - Provide ONLY a direct answer to the user's question
    - DO NOT ask follow-up questions under any circumstances
    - DO NOT suggest additional topics or related questions
    - End your response when you have answered the question
    - If information is missing, state what you cannot answer and stop there

    <context>
    {context}
    </context>

    User Question:
    {question}

    Your Answer:
    """
    )

def format_history(chat_history: List[str]) -> str:
    """Format chat history, keeping only last 2 exchanges to prevent context overflow"""
    return "\n".join(chat_history[-4:])

def retrieve(state: State):
    """Retrieve relevant documents"""
    retrieved_docs = st.session_state.hybrid_retriever_reranking.invoke(state["question"], k=2)
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate response using the LLM"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    history_text = format_history(state.get("chat_history", []))
    
    if history_text:
        full_context = f"Retrieved Information:\n{docs_content}\n\nRecent Conversation History:\n{history_text}"
    else:
        full_context = f"Retrieved Information:\n{docs_content}"
    
    messages = st.session_state.prompt.invoke({"question": state["question"], "context": full_context})
    response = st.session_state.llm.invoke(messages)
    
    updated_history = state.get("chat_history", []) + [
        f"User: {state['question']}", f"Assistant: {response}"
    ]
    
    return {
        "answer": response,
        "chat_history": updated_history
    }

def create_graph():
    """Create the LangGraph workflow"""
    graph_builder = StateGraph(State)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge("retrieve", "generate")
    
    return graph_builder.compile()

def main():
    # Header
    st.markdown("<h1 class='main-header'>🎓 NOVA IMS AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This AI assistant helps NOVA IMS students with course-related questions. "
            "Ask about course structures, requirements, or any academic information."
        )
        
        st.header("🛠️ Settings")
        show_sources = st.checkbox("Show source documents", value=False)
        show_context = st.checkbox("Show retrieved context", value=False)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.header("📊 System Status")
        if 'initialized' in st.session_state:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ Initializing...")
    
    # Initialize components
    if 'initialized' not in st.session_state:
        try:
            with st.spinner("🚀 Initializing AI Assistant..."):
                # Initialize model
                st.session_state.llm = initialize_model()
                
                # Initialize retrievers with the loaded documents
                st.session_state.hybrid_retriever_reranking = initialize_retrievers(documents_chunked_with_ids_and_metadata)
                
                # Initialize prompt
                st.session_state.prompt = create_prompt_template()
                
                # Create graph
                st.session_state.graph = create_graph()
                
                st.session_state.initialized = True
            
            st.success("✅ AI Assistant ready!")
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if enabled and available
            if show_sources and "sources" in message:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"**Document {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Metadata: {doc.metadata}")
    
    # Chat input
    if prompt_input := st.chat_input("Ask me about NOVA IMS courses..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Create state for the graph
                    state = {
                        "question": prompt_input,
                        "chat_history": st.session_state.chat_history
                    }
                    
                    # Get response from the graph
                    result = st.session_state.graph.invoke(state)
                    
                    # Update session state
                    st.session_state.chat_history = result.get("chat_history", [])
                    
                    # Display response
                    st.markdown(result["answer"])
                    
                    # Show context if enabled
                    if show_context and "context" in result:
                        with st.expander("🔍 Retrieved Context"):
                            for i, doc in enumerate(result["context"], 1):
                                st.markdown(f"**Context {i}:**")
                                st.text(doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.caption(f"Metadata: {doc.metadata}")
                    
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result["answer"]
                    }
                    
                    if "context" in result:
                        assistant_message["sources"] = result["context"]
                    
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
    
    # Example questions
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    
    example_questions = [
        "How is the Business Intelligence postgraduate curriculum structured in the first semester?",
        "What are the mandatory courses in the Data Science program?",
        "Can you tell me about the elective courses available?",
        "What are the requirements for the Information Management degree?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {question}", key=f"example_{i}"):
                # Add the question to chat input
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Process the question
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        try:
                            state = {
                                "question": question,
                                "chat_history": st.session_state.chat_history
                            }
                            
                            result = st.session_state.graph.invoke(state)
                            st.session_state.chat_history = result.get("chat_history", [])
                            
                            st.markdown(result["answer"])
                            
                            assistant_message = {
                                "role": "assistant", 
                                "content": result["answer"]
                            }
                            
                            if "context" in result:
                                assistant_message["sources"] = result["context"]
                            
                            st.session_state.messages.append(assistant_message)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing example question: {str(e)}")
                
                st.rerun()

if __name__ == "__main__":
    main()





















