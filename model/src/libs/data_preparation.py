import os 
import re
from libs.settings import data_catalog as dc
from libs import data_extraction as de
import pickle
from typing import Dict, Optional, List


def clean_text_documents(
    data: Dict[str, Dict],
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None,
    words_to_remove: Optional[List[str]] = None,
    words_to_deduplicate: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Cleans text documents by:
    1. Removing specified words/phrases completely.
    2. Ensuring only a single occurrence remains for repetitive words.

    Parameters:
    - data (dict): Dictionary where keys are filenames and values contain 'text' and 'metadata'.
    - course_names_to_include (list, optional): Filter by specific course names (case insensitive).
    - doc_types_to_include (list, optional): Filter by document types ('teaching_staff', 'study_plan', 'main_info').
    - words_to_remove (list, optional): List of words or phrases to remove entirely.
    - words_to_deduplicate (list, optional): List of words where only one occurrence should remain.

    Returns:
    - dict: A dictionary with cleaned text and original metadata.
    """
    cleaned_data = {}

    remove_pattern = None
    if words_to_remove:
        remove_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'

    for filename, doc in data.items():
        course_name = doc["metadata"].get("course_name", "").lower()
        doc_type = doc["metadata"].get("doc_type", "").lower()

        # Filter by course name
        if course_names_to_include and course_name not in [name.lower() for name in course_names_to_include]:
            continue

        # Filter by doc type
        if doc_types_to_include and doc_type not in [dt.lower() for dt in doc_types_to_include]:
            continue

        cleaned_text = doc["text"]

        # Remove specified phrases
        if remove_pattern:
            cleaned_text = re.sub(remove_pattern, '', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Deduplicate repeated phrases
        if words_to_deduplicate:
            for word in words_to_deduplicate:
                word_pattern = rf'\b({re.escape(word)})(?:\s+\1)+\b'
                cleaned_text = re.sub(word_pattern, r'\1', cleaned_text, flags=re.IGNORECASE)

        cleaned_data[filename] = {
            "text": cleaned_text.strip(),
            "metadata": doc["metadata"]
        }

    return cleaned_data


###### Function to identify large documents ######
def get_large_documents(text_data, min_word_count):
    """
    Identifies documents that exceed a given word count threshold.

    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    - min_word_count (int): Minimum number of words a document must have to be considered "large".

    Returns:
    - list: Filenames of documents that exceed the word count threshold.
    """
    large_documents = [filename for filename, content in text_data.items() if len(content.split()) > min_word_count]
    return large_documents


