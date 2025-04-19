import os 
import re
from libs.settings import data_catalog as dc
from libs import data_extraction as de
import pickle
from typing import Dict, Optional, List


def clean_and_save_all_programs_textfiles(output_directory):
    """
    Loads raw text dictionaries from pickle files, cleans them using provided logic, 
    and saves the cleaned dictionaries back to pickle format in the output directory.

    :param save_directory: Directory containing raw pickle files.
    :param output_directory: Directory to save the cleaned pickle files.
    :param clean_text_documents_func: Function to clean text documents.
    """

    os.makedirs(output_directory, exist_ok=True)
    save_directory= dc.HP_PATH_RAW_EXTRACTED_DOCS_DICTS

    all_programs_dict_textfiles_raw = de.load_all_programs_dict_textfiles(save_directory)

    for outer_key, inner_dict in all_programs_dict_textfiles_raw.items():
        # === Define custom cleaning parameters per outer_key ===
        if outer_key == "dict_bs_teaching_staff_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["know", "more", "apply", "here", "Education"],
                words_to_deduplicate=["Teaching Staff"]
            )
        elif outer_key == "dict_bs_studyplan_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["Programs", "resumo do conteudo da tabela", "Loading...", "Education", "modal item",
                     "card item"]
            )
        elif outer_key == "dict_bs_maininfo_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["Programs", "resumo do conteudo da tabela", "Loading...", "Education", "caption text",
                     "card item", "modal item", "Apply here", "Know more"],
                words_to_deduplicate=["Data Science", "Information Management", "Information Systems", "Who is it for?"]
            )
        elif outer_key == "dict_pm_teaching_staff_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["know", "more", "apply", "here"], 
                words_to_deduplicate=["faculty"]
            )
        elif outer_key == "dict_pm_studyplan_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["loading", "... modal item card item",
                                "resumo do conteudo da tabela"],
                words_to_deduplicate=["Study plan"]
            )
        elif outer_key == "dict_pm_maininfo_raw":
            cleaned_inner_dict = clean_text_documents(
                inner_dict,
                words_to_remove=["en", "To apply,", "click", "here",
                    "resumo do conteudo da tabela",
                    "Loading...","...", "resumo do conteudo da tabela", "Know more",
                    "modal item", "card item", "caption text", "Value", "Annual Prize", "Program", "Unit",
                    "Curricular", "Programs", "Education", "Postgraduate Programs and Master Degree Programs"
                    ]
            )

        # === Save each cleaned dictionary ===
        output_path = os.path.join(output_directory, f"{outer_key}_cleaned.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(cleaned_inner_dict, f)

        print(f"Saved cleaned dictionary: {output_path}")


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
        remove_pattern = r'(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')'

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


