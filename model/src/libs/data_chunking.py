import os
from langchain_core.documents import Document
import pickle
import re
from typing import List, Dict
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from nltk import sent_tokenize


def save_documents_to_disk(documents: List[Document], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_disk(input_path: str) -> List[Document]:
    with open(input_path, "rb") as f:
        return pickle.load(f)


def process_chunking_documents(
                                input_folder: str,
                                output_folder: str,
                                max_tokens_by_type: dict,
                            ) -> None:
    
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".pkl"):
            continue

        input_path = os.path.join(input_folder, file_name)
        with open(input_path, "rb") as f:
            data_dict = pickle.load(f)

        if "teachingstaff" in file_name.lower():
            max_tokens = max_tokens_by_type.get("teachingstaff")
            documents = chunk_teaching_staff_documents(data_dict, max_tokens)
        elif "studyplan" in file_name.lower():
            max_tokens = max_tokens_by_type.get("studyplan")
            documents = chunk_study_plan_documents(data_dict, max_tokens)
        elif "maininfo" in file_name.lower():
            max_tokens = max_tokens_by_type.get("maininfo")
            documents = chunk_maininfo_documents(data_dict, max_tokens)
        else:
            continue

        output_path = os.path.join(output_folder, file_name)
        save_documents_to_disk(documents, output_path)



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
    doc_dict: Dict[str, str],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 0
) -> List[Document]:
    """
    Main entry point. Accepts a dictionary {doc_name: text} and returns chunked Documents.
    """
    all_docs = []

    for name, text in doc_dict.items():
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping empty or invalid text for: {name}")
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
                all_docs.append(Document(page_content=chunk, metadata={"source": name}))

        except Exception as e:
            print(f"❌ Error processing teaching staff in '{name}': {e}")

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
    doc_dict: Dict[str, str],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 50
) -> List[Document]:
    
    all_docs = []

    if not isinstance(doc_dict, dict):
        raise ValueError("❌ Input must be a dictionary with document names and text.")

    for name, text in doc_dict.items():
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping empty or invalid text for: {name}")
            continue
        try:
            chunks = smart_chunk_by_semester(
                text,
                max_tokens=max_tokens,
                chunk_overlap=chunk_overlap,
                overlap_size=overlap_size
            )
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata={"source": name}))
        except Exception as e:
            print(f"❌ Error chunking document '{name}': {e}")
    
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
def chunk_maininfo_documents(
    doc_dict: Dict[str, str],
    max_tokens: int = 512,
    chunk_overlap: bool = False,
    overlap_size: int = 50
) -> List[Document]:
    chunked_documents = []

    for course_name, text in doc_dict.items():
        lines = text.split("\n")
        sections = []
        current_section = {"header": None, "content": ""}

        for line in lines:
            if is_potential_section_header(line):
                if current_section["header"] or current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"header": line.strip(), "content": ""}
            else:
                current_section["content"] += line.strip() + " "

        if current_section["header"] or current_section["content"].strip():
            sections.append(current_section)

        for section in sections:
            full_text = (section["header"] + "\n" if section["header"] else "") + section["content"]
            sentences = sent_tokenize(full_text)

            buffer = []
            token_count = 0
            i = 0

            while i < len(sentences):
                sentence = sentences[i]
                sentence_tokens = count_tokens(sentence)

                if sentence_tokens > max_tokens:
                    raise ValueError(
                        f"❌ Sentence too large to fit in a chunk (size={sentence_tokens} tokens, max={max_tokens}):\n{sentence}"
                    )

                if token_count + sentence_tokens <= max_tokens:
                    buffer.append(sentence)
                    token_count += sentence_tokens
                    i += 1
                else:
                    chunk_text = " ".join(buffer).strip()
                    chunked_documents.append(
                        Document(page_content=chunk_text, metadata={"source": course_name})
                    )

                    # Handle overlap
                    if chunk_overlap:
                        overlap_buffer = []
                        overlap_token_count = 0
                        for s in reversed(buffer):
                            s_tokens = count_tokens(s)
                            if overlap_token_count + s_tokens <= overlap_size:
                                overlap_buffer.insert(0, s)
                                overlap_token_count += s_tokens
                            else:
                                break
                        buffer = overlap_buffer
                        token_count = overlap_token_count
                    else:
                        buffer = []
                        token_count = 0

            if buffer:
                chunk_text = " ".join(buffer).strip()
                chunked_documents.append(
                    Document(page_content=chunk_text, metadata={"source": course_name})
                )

    return chunked_documents

# === Helper Function for Main Info Chunking ===
def is_potential_section_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return (
        len(line) < 80 and (
            line.isupper() or
            line.istitle() or
            bool(re.match(r"^\d+[\).\s]", line))
        )
    )