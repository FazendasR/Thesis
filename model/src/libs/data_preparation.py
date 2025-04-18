import os 
import re
from libs.settings import data_catalog as dc
from libs import data_extraction as de
import pickle


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

    all_programs_dict_textfiles_raw = de.load_all_programs_dict_textfiles_raw(save_directory)

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


def clean_text_documents(text_data, words_to_remove=None, words_to_deduplicate=None):
    """
    Cleans text documents by:
    1. Removing specified words/phrases completely.
    2. Ensuring only a single occurrence remains for repetitive words.

    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    - words_to_remove (list, optional): List of words or phrases to remove entirely.
    - words_to_deduplicate (list, optional): List of words where only one occurrence should remain.

    Returns:
    - dict: A dictionary with cleaned text.
    """
    cleaned_text_data = {}

    # Compile patterns only if the respective lists are provided
    remove_pattern = None
    if words_to_remove:
        # Use exact phrase matching instead of word boundaries
        remove_pattern = r'(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')'

    for filename, text in text_data.items():
        cleaned_text = text

        # Remove words and phrases completely
        if remove_pattern:
            cleaned_text = re.sub(remove_pattern, '', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Clean extra spaces

        # Remove duplicate occurrences of specific words
        if words_to_deduplicate:
            for word in words_to_deduplicate:
                word_pattern = rf'\b{re.escape(word)}(?:\s+{re.escape(word)})+\b'
                cleaned_text = re.sub(word_pattern, word, cleaned_text, flags=re.IGNORECASE)

        # Store cleaned text
        cleaned_text_data[filename] = cleaned_text.strip()

    return cleaned_text_data


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


