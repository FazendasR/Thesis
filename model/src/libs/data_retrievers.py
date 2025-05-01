from langchain.retrievers import BM25Retriever, TFIDFRetriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 
from langchain.retrievers import ContextualCompressionRetriever

def load_sparse_retriever(retriever_type, documents_chunked, top_k=4):
    """
    Load a sparse retriever based on the chosen type.
    
    Args:
        retriever_type (str): Type of retriever ('TF-IDF' or 'BM25').
        documents_chunked (list): List of documents to be passed to the retriever.
        k (int): Number of documents to retrieve (default is 4).
        
    Returns:
        retriever: A sparse retriever (either BM25 or TF-IDF).
    """
    if retriever_type == 'BM25':
        return BM25Retriever.from_documents(documents_chunked, k=top_k)
    elif retriever_type == 'TF-IDF':
        return TFIDFRetriever.from_documents(documents_chunked, k=top_k)
    else:
        raise ValueError("Invalid retriever type. Choose either 'BM25' or 'TF-IDF'.")
    


def load_vector_retriever(collection_name="documents_without_metadata_embedded", top_k=4, persist_directory="./chroma_langchain_db"):
    """
    Load a vector retriever from a Chroma vector store using a HuggingFace embedding model.
    
    Args:
        collection_name (str): The name of the collection in the Chroma vector store (default is 'documents_without_metadata_embedded').
        persist_directory (str): The directory to persist the Chroma vector store (default is './chroma_langchain_db').

    Returns:
        ChromaRetriever: A retriever based on the Chroma vector store.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Important for BGE
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return chroma_retriever


def load_hybrid_retriever(sparse_retriever, vector_retriever, weight_sparse=0.5, weight_vector=0.5):
    """
    Load a hybrid retriever that combines a sparse retriever (BM25/TF-IDF) and a vector retriever (Chroma).
    
    Args:
        sparse_retriever (Retriever): A sparse retriever, such as BM25Retriever or TFIDFRetriever.
        vector_retriever (Retriever): A vector retriever, such as ChromaRetriever.
        weight_sparse (float): The weight for the sparse retriever (default is 0.5).
        weight_vector (float): The weight for the vector retriever (default is 0.5).
        top_k (int): The number of top results to return (default is 4).
        
    Returns:
        List: A list of the top `top_k` hybrid retriever results.
    """
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, vector_retriever],
        weights=[weight_sparse, weight_vector]
    )
    
    return ensemble_retriever

def get_reranking(base_retriever, top_n=4):
    """
    Use a cross-encoder reranker to re-rank the top_n documents retrieved by the base retriever.
    
    Args:
        base_retriever (Retriever): The base retriever that retrieves the initial top documents.
        top_n (int): The number of top documents to retrieve and re-rank (default is 4).
        
    Returns:
        List: Re-ranked top_n documents.
    """
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,  
    )
    return compression_retriever
