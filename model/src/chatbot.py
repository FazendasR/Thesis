"""
NOVA Information Management School Chatbot Module

This module contains the chatbot implementation for answering student questions
about courses and professors using hybrid retrieval and LLM generation.
"""

import os
import pickle
import re
import torch
from typing import List
from typing_extensions import TypedDict

# LangChain imports
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangGraph imports
from langgraph.graph import START, StateGraph

# Local imports (adjust these paths based on your project structure)
try:
    from libs import data_handeling as dh
    from libs.settings import data_catalog as dc
    from libs import data_retrievers as dr
except ImportError as e:
    print(f"Warning: Could not import local libraries: {e}")
    print("Make sure your working directory includes the libs folder")


class State(TypedDict):
    """State structure for the chatbot conversation flow"""
    question: str
    context: List[Document]
    answer: str
    chat_history: List[str]


class NOVAChatbot:
    """
    NOVA Information Management School Chatbot
    
    A chatbot designed to answer student questions about courses and professors
    using hybrid retrieval (TF-IDF + vector search) and LLM generation.
    """
    
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct", 
                 collection_name="parent_documents_with_ids_and_metadata_embedded_v2"):
        """
        Initialize the chatbot with model and retrievers
        
        Args:
            model_id (str): Hugging Face model identifier
            collection_name (str): Chroma collection name for vector retrieval
        """
        self.model_id = model_id
        self.collection_name = collection_name
        self.documents_chunked = None
        self.tfidf_retriever = None
        self.chroma_retriever = None
        self.hybrid_retriever = None
        self.hybrid_retriever_reranking = None
        self.llm = None
        self.prompt = None
        self.graph = None
        
        # Initialize the chatbot components
        self._setup_working_directory()
        self._load_documents()
        self._setup_retrievers()
        self._setup_llm()
        self._setup_prompt()
        self._setup_graph()
    
    def _setup_working_directory(self):
        """Setup the working directory to access model files"""
        try:
            # Move to Thesis directory (two levels up)
            os.chdir(os.path.abspath(os.path.join("..", "..")))
            
            # Move to model/src if it exists
            model_dir = os.path.join(os.getcwd(), "model", "src")
            if os.path.exists(model_dir):
                os.chdir(model_dir)
            
            print(f"Working directory set to: {os.getcwd()}")
        except Exception as e:
            print(f"Warning: Could not set working directory: {e}")
    
    def _load_documents(self):
        """Load and process documents for retrieval"""
        try:
            self.documents_chunked = dh.load_documents_from_pickle(
                dc.DOCUMENTS_CHUNKED_WITH_IDS_AND_METADATA
            )
            self.documents_chunked = self._move_id_to_metadata(self.documents_chunked)
            print(f"Loaded {len(self.documents_chunked)} documents")
        except Exception as e:
            print(f"Error loading documents: {e}")
            self.documents_chunked = []
    
    def _move_id_to_metadata(self, documents):
        """Move document IDs to metadata for proper handling"""
        for doc in documents:
            if hasattr(doc, 'id') and doc.id is not None:
                doc.metadata['id'] = doc.id
        return documents
    
    def _setup_retrievers(self):
        """Setup TF-IDF, vector, and hybrid retrievers"""
        try:
            self.tfidf_retriever = dr.load_sparse_retriever(
                retriever_type="TF-IDF", 
                documents_chunked=self.documents_chunked, 
                top_k=5
            )
            
            self.chroma_retriever = dr.load_vector_retriever(
                collection_name=self.collection_name, 
                top_k=5
            )
            
            self.hybrid_retriever = dr.load_hybrid_retriever(
                self.tfidf_retriever, 
                self.chroma_retriever, 
                weight_sparse=0.5, 
                weight_vector=0.5
            )
            
            self.hybrid_retriever_reranking = dr.get_reranking(
                self.hybrid_retriever, 
                top_n=2
            )
            
            print("Retrievers setup completed")
        except Exception as e:
            print(f"Error setting up retrievers: {e}")
    
    def _setup_llm(self):
        """Setup the Hugging Face LLM pipeline"""
        try:
            print(f"Initializing model: {self.model_id}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            
            # Create text generation pipeline
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
            
            # Setup LangChain LLM with callbacks
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = HuggingFacePipeline(
                pipeline=text_generation_pipeline,
                model_id=self.model_id,
                callbacks=callback_manager,
            )
            
            print("Hugging Face model successfully integrated into LangChain")
        except Exception as e:
            print(f"Error setting up LLM: {e}")
    
    def _setup_prompt(self):
        """Setup the prompt template for the chatbot"""
        self.prompt = PromptTemplate(
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
    
    def _retrieve(self, state: State):
        """Retrieve relevant documents for the question"""
        retrieved_docs = self.hybrid_retriever_reranking.invoke(state["question"], k=2)
        return {"context": retrieved_docs}
    
    def _format_history(self, chat_history: List[str]) -> str:
        """Format chat history, keeping only last 2 exchanges to prevent context overflow"""
        return "\n".join(chat_history[-4:])  # Last 2 user-assistant pairs
    
    def _generate(self, state: State):
        """Generate answer using LLM based on context and question"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        history_text = self._format_history(state.get("chat_history", []))
        
        # Format context with clear separation
        if history_text:
            full_context = f"Retrieved Information:\n{docs_content}\n\nRecent Conversation History:\n{history_text}"
        else:
            full_context = f"Retrieved Information:\n{docs_content}"
        
        messages = self.prompt.invoke({"question": state["question"], "context": full_context})
        response = self.llm.invoke(messages)
        
        updated_history = state.get("chat_history", []) + [
            f"User: {state['question']}", f"Assistant: {response}"
        ]
        
        return {
            "answer": response,
            "chat_history": updated_history
        }
    
    def _setup_graph(self):
        """Setup the LangGraph workflow"""
        try:
            graph_builder = StateGraph(State)
            
            # Add nodes explicitly
            graph_builder.add_node("retrieve", self._retrieve)
            graph_builder.add_node("generate", self._generate)
            
            # Add edges
            graph_builder.add_edge(START, "retrieve")
            graph_builder.add_edge("retrieve", "generate")
            
            self.graph = graph_builder.compile()
            print("Graph workflow setup completed")
        except Exception as e:
            print(f"Error setting up graph: {e}")
    
    def ask(self, question: str, chat_history: List[str] = None) -> dict:
        """
        Ask a question to the chatbot
        
        Args:
            question (str): The user's question
            chat_history (List[str], optional): Previous conversation history
            
        Returns:
            dict: Response containing answer and updated chat history
        """
        if not self.graph:
            return {"error": "Chatbot not properly initialized"}
        
        try:
            initial_state = {
                "question": question,
                "context": [],
                "answer": "",
                "chat_history": chat_history or []
            }
            
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result["answer"],
                "chat_history": result["chat_history"],
                "context": [doc.page_content for doc in result["context"]]
            }
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}
    
    def get_answer_only(self, question: str, chat_history: List[str] = None) -> str:
        """
        Get only the answer text from the chatbot
        
        Args:
            question (str): The user's question
            chat_history (List[str], optional): Previous conversation history
            
        Returns:
            str: The chatbot's answer
        """
        result = self.ask(question, chat_history)
        if "error" in result:
            return f"Error: {result['error']}"
        return result["answer"]


# Convenience function for quick initialization
def create_chatbot(model_id="meta-llama/Llama-3.2-3B-Instruct", 
                   collection_name="parent_documents_with_ids_and_metadata_embedded_v2"):
    """
    Create and return a NOVA chatbot instance
    
    Args:
        model_id (str): Hugging Face model identifier
        collection_name (str): Chroma collection name
        
    Returns:
        NOVAChatbot: Initialized chatbot instance
    """
    return NOVAChatbot(model_id=model_id, collection_name=collection_name)


if __name__ == "__main__":
    # Example usage when running the module directly
    print("Initializing NOVA Chatbot...")
    chatbot = create_chatbot()
    
    print("Chatbot initialized successfully. You can now use the chatbot instance to ask questions.")