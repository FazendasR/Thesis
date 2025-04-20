import os
from langchain_core.documents import Document
import pickle
import re
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from nltk import sent_tokenize
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


# === Tokenizer Setup ===
def load_tokenizer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> PreTrainedTokenizerFast:
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load tokenizer: {e}")

tokenizer = load_tokenizer()

def count_tokens(text: str) -> int:
    try:
        return len(tokenizer.encode(text, truncation=False))
    except Exception as e:
        print(f"⚠️ Token counting failed for text: {text[:50]}... \nError: {e}")
        return float("inf")
    

# === Chunking Function for Teaching Staff ===
def chunk_teaching_staff_documents(
    data: Dict[str, Dict],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 0,
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None,
    include_metadata: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Chunk teaching staff documents into LangChain Documents with optional filters and metadata injection.

    Parameters:
        data: Dict of {filename: {'text': ..., 'metadata': {...}}}
        max_tokens: Max token length for chunks
        chunk_overlap: Whether to overlap chunks
        overlap_size: Overlap size (in blocks)
        course_names_to_include: Optional filter by course name
        doc_types_to_include: Optional filter by doc_type
        include_metadata: If True, include existing metadata
        extra_metadata: Optional extra metadata to attach to each chunk

    Returns:
        List of LangChain Document objects

    Example: 
        chunked_docs = chunk_teaching_staff_documents(
        data=my_data,
        max_tokens=512,
        chunk_overlap=False,
        overlap_size=0,
        doc_types_to_include=["teaching_staff"],
        course_names_to_include=["Data Science"],
        extra_metadata={"chunked_by": "teaching_staff_function", "version": "v1.2"},
        include_metadata=True
)
    """
    all_docs = []

    for file_name, file_data in data.items():
        text = file_data.get("text", "")
        metadata = file_data.get("metadata", {})

        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping empty or invalid text for: {file_name}")
            continue

        course_name = metadata.get("course_name", "")
        doc_type = metadata.get("doc_type", "")

        if course_names_to_include and course_name not in course_names_to_include:
            continue
        if doc_types_to_include and doc_type not in doc_types_to_include:
            continue

        try:
            blocks = extract_teaching_staff_blocks(text)
            grouped_chunks = group_blocks_by_token_limit(
                blocks,
                max_token_size=max_tokens,
                chunk_overlap=chunk_overlap,
                overlap_size=overlap_size
            )

            for chunk in grouped_chunks:
                doc_metadata = {"source": file_name}

                if include_metadata:
                    doc_metadata.update(metadata)

                if extra_metadata:
                    doc_metadata.update(extra_metadata)

                all_docs.append(Document(page_content=chunk, metadata=doc_metadata))

        except Exception as e:
            print(f"❌ Error processing teaching staff in '{file_name}': {e}")

    return all_docs


# === Helper Functions for Teaching Staff Chunking ===
def extract_teaching_staff_blocks(text: str) -> List[str]:
    """
    Extracts blocks of teaching staff information in the format:
    Name\nTitle\nEmail
    """
    pattern = r"([A-Z][a-zA-Zçéàèíóõâêã]+\s+[A-Z][a-zA-Zçéàèíóõâêã]+.*?)\s([\w\.\s]+?)\s([\w\.-]+@novaims\.unl\.pt)"
    matches = re.findall(pattern, text)
    return [f"{name}\n{title}\n{email}\n\n" for name, title, email in matches]

def group_blocks_by_token_limit(
    blocks: List[str],
    max_token_size: int,
    chunk_overlap: bool = False,
    overlap_size: int = 0
) -> List[str]:
    """
    Groups extracted blocks into text chunks not exceeding max_token_size tokens.
    Optionally includes overlapping blocks between consecutive chunks.
    """
    grouped_chunks = []
    current_chunk = []
    current_token_count = 0
    i = 0

    while i < len(blocks):
        block = blocks[i]
        token_count = count_tokens(block)

        if token_count > max_token_size:
            raise ValueError(
                f"❌ Block too large to fit in a chunk (size={token_count} tokens, max={max_token_size}):\n{block}"
            )

        if current_token_count + token_count > max_token_size:
            if current_chunk:
                grouped_chunks.append("".join(current_chunk).strip())

                if chunk_overlap and overlap_size > 0:
                    # Keep last N blocks as overlap
                    current_chunk = current_chunk[-overlap_size:]
                    current_token_count = sum(count_tokens(b) for b in current_chunk)
                else:
                    current_chunk = []
                    current_token_count = 0
        else:
            current_chunk.append(block)
            current_token_count += token_count
            i += 1

    if current_chunk:
        grouped_chunks.append("".join(current_chunk).strip())

    return grouped_chunks
# === Chunking Function for Study Plan ===
def chunk_study_plan_documents(
    data: Dict[str, Dict],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 0,
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None,
    include_metadata: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    all_docs = []

    for file_name, file_data in data.items():
        text = file_data.get("text", "")
        metadata = file_data.get("metadata", {})

        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping empty or invalid text for: {file_name}")
            continue

        course_name = metadata.get("course_name", "")
        doc_type = metadata.get("doc_type", "")

        if course_names_to_include and course_name not in course_names_to_include:
            continue
        if doc_types_to_include and doc_type not in doc_types_to_include:
            continue

        try:
            chunks = smart_chunk_by_semester(
                text,
                max_tokens=max_tokens,
                chunk_overlap=chunk_overlap,
                overlap_size=overlap_size
            )
            for chunk in chunks:
                doc_metadata = {"source": file_name}

                if include_metadata:
                    doc_metadata.update(metadata)

                if extra_metadata:
                    doc_metadata.update(extra_metadata)

                all_docs.append(Document(page_content=chunk, metadata=doc_metadata))

        except Exception as e:
            print(f"❌ Error chunking study plan in '{file_name}': {e}")

    return all_docs


# === Helper Functions for Study Plan Chunking ===
def split_to_token_limit(text: str, max_tokens: int, overlap: bool = False, overlap_size: int = 50) -> List[str]:
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        chunks.append(chunk_text)
        start = end - overlap_size if overlap else end
    return chunks

def smart_chunk_by_semester(text: str, max_tokens: int = 512, chunk_overlap: bool = False, overlap_size: int = 50) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    pattern = re.compile(r"(\d+\s?(?:st|nd|rd|th) year - (?:Fall|Spring) Semester)", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if not matches:
        print("⚠️ No semester headers found — chunking whole text instead.")
        return split_to_token_limit(text, max_tokens, overlap=chunk_overlap, overlap_size=overlap_size)

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if count_tokens(chunk_text) <= max_tokens:
            chunks.append(chunk_text)
        else:
            chunks.extend(split_to_token_limit(chunk_text, max_tokens, overlap=chunk_overlap, overlap_size=overlap_size))

    return chunks

# === Chunking Function for Main Info ===
def chunk_main_info_documents(
    data: Dict[str, Dict],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 50,
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None,
    include_metadata: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    documents = []
    text_splitter = get_text_splitter(max_tokens, overlap_size if chunk_overlap else 0)

    for filename, content_dict in data.items():
        text = content_dict.get("text", "")
        metadata = content_dict.get("metadata", {})

        course_name = metadata.get("course_name", "")
        doc_type = metadata.get("doc_type", "")

        if course_names_to_include and course_name not in course_names_to_include:
            continue
        if doc_types_to_include and doc_type not in doc_types_to_include:
            continue
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping empty or invalid text in: {filename}")
            continue

        # Split the full text using RecursiveCharacterTextSplitter
        chunks = text_splitter.split_text(text)

        # Create Document objects for each chunk
        for chunk in chunks:
            doc_metadata = {"source": filename}
            if include_metadata:
                doc_metadata.update(metadata)
                if extra_metadata:
                    doc_metadata.update(extra_metadata)
            
            documents.append(Document(page_content=chunk, metadata=doc_metadata))

    return documents

# === RecursiveCharacterTextSplitter Setup ===
def get_text_splitter(max_tokens: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,  # Max number of characters per chunk
        chunk_overlap=chunk_overlap,  # Number of overlapping characters between chunks
        length_function=len,  # Function to compute the length of each chunk (by character count)
        separators=["\n", "\n\n", ".", "!", "?"]  # Prefer breaking on these separators for better semantic breaks
    )