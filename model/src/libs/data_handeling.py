import pickle
import os
from typing import Dict, List, Optional
from libs.settings import data_catalog as dc
from libs import data_preparation as dp

def _save_dict_program_textfiles_to_pickle(data_dict: Dict[str, Dict], output_file_name: str, output_folder: str):
    '''
        Save a dictionary of dictionaries to a pickle file.

    '''
    os.makedirs(output_folder, exist_ok=True)  
    output_path = os.path.join(output_folder, output_file_name)

    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"✅ Saved: {output_path}")

def load_pickle_to_dict(pickle_file_path: str) -> Dict[str, Dict]:
    if not os.path.exists(pickle_file_path):
        raise FileNotFoundError(f"❌ File not found: {pickle_file_path}")

    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)
    return data

def print_text_context_from_program_dicts(
                                            data: Dict[str, Dict],
                                            course_names_to_include: Optional[List[str]] = None,
                                            doc_types_to_include: Optional[List[str]] = None
                                        ):
    """
    Prints the text content of the documents based on the specified filters.

    Parameters:
    - data (dict): Dictionary with filenames as keys and dicts containing 'text' and 'metadata'.
    - course_names_to_include (list, optional): Filter by specific course names (case insensitive).
    - doc_types_to_include (list, optional): Filter by document types ('teaching_staff', 'study_plan', 'main_info').
    """
    filtered_data = {}

    for filename, doc in data.items():
        course_name = doc["metadata"].get("course_name", "").lower()
        doc_type = doc["metadata"].get("doc_type", "").lower()

        # Filter by course name
        if course_names_to_include and course_name not in [name.lower() for name in course_names_to_include]:
            continue

        # Filter by doc type
        if doc_types_to_include and doc_type not in [dt.lower() for dt in doc_types_to_include]:
            continue

        filtered_data[filename] = doc

    if not filtered_data:
        print("⚠️ No documents matched the filters.")
        return

    # Print the text content of the filtered documents
    for filename, doc in filtered_data.items():
        print(f"\n--- Document: {filename} ---")
        print(f"Course Name: {doc['metadata'].get('course_name')}")
        print(f"Document Type: {doc['metadata'].get('doc_type')}")
        print("\nText Content:")
        print(doc["text"])
        print("\n" + "-"*50)


def create_dict_programs_cleaned():
    """
    Load raw bachelor and postgrad data, apply custom cleaning rules per document type,
    and save cleaned dictionaries preserving original structure.
    """
    # Load raw data
    bachelors_data_raw = load_pickle_to_dict(dc.BACHELORS_DATA_RAW)
    postgradmasters_data_raw = load_pickle_to_dict(dc.POSTGRAD_AND_MASTERS_DATA_RAW)

    # === Define cleaning rules for each document type ===
    bachelors_rules = {
        "teaching_staff": {
            "words_to_remove": ["know", "more", "apply", "here", "Education"],
            "words_to_deduplicate": ["Teaching Staff"]
        },
        "study_plan": {
            "words_to_remove": ["Programs", "resumo do conteudo da tabela", "Loading...", "Education", "modal item", "card item"],
            "words_to_deduplicate": []
        },
        "main_info": {
            "words_to_remove": [
                "Programs", "resumo do conteudo da tabela", "Loading...", "Education", "caption text",
                "card item", "modal item", "Apply here", "Know more"
            ],
            "words_to_deduplicate": ["Data Science", "Information Management", "Information Systems", "Who is it for?"]
        }
    }

    postgrad_rules = {
        "teaching_staff": {
            "words_to_remove": ["know", "more", "apply", "here"],
            "words_to_deduplicate": ["faculty"]
        },
        "study_plan": {
            "words_to_remove": ["loading", "... modal item card item", "resumo do conteudo da tabela"],
            "words_to_deduplicate": ["Study plan"]
        },
        "main_info": {
            "words_to_remove": [
                "en", "To apply,", "click", "here", "resumo do conteudo da tabela", "Loading...", "...", "Know more",
                "modal item", "card item", "caption text", "Value", "Annual Prize", "Program", "Unit", "Curricular",
                "Programs", "Education", "Postgraduate Programs and Master Degree Programs"
            ],
            "words_to_deduplicate": []
        }
    }

    # === Clean raw data ===
    def clean_all_doc_types(raw_data: Dict[str, Dict], rules: Dict[str, Dict]) -> Dict[str, Dict]:
        cleaned = {}
        for doc_type, params in rules.items():
            cleaned_part = dp.clean_text_documents(
                data=raw_data,
                doc_types_to_include=[doc_type],
                words_to_remove=params["words_to_remove"],
                words_to_deduplicate=params["words_to_deduplicate"]
            )
            cleaned.update(cleaned_part)
        return cleaned

    bachelors_data_cleaned = clean_all_doc_types(bachelors_data_raw, bachelors_rules)
    postgradmasters_data_cleaned = clean_all_doc_types(postgradmasters_data_raw, postgrad_rules)

    # === Save cleaned dicts ===
    _save_dict_program_textfiles_to_pickle(
        data_dict=bachelors_data_cleaned,
        output_file_name="bachelors_textfiles_cleaned.pkl",
        output_folder=dc.PATH_CLEANED_DOCS_DICTS  
    )

    _save_dict_program_textfiles_to_pickle(
        data_dict=postgradmasters_data_cleaned,
        output_file_name="postgradmasters_textfiles_cleaned.pkl",
        output_folder=dc.PATH_CLEANED_DOCS_DICTS 
    )



# ==== Create Dictionary of Programs Raw Text Files ====
def create_dict_programs_raw():
    '''
        Create a dictionary of dictionaries for all programs.
        Each dictionary represents a program's document type (teaching_staff, study_plan, main_info) files.
    '''
    folder_paths = [
        r"../../data/Webscrapping/bachelor_degree/",
        r"../../data/Webscrapping/postgraduate_master_degrees/teachingstaff",
        r"../../data/Webscrapping/postgraduate_master_degrees/studyplan",
        r"../../data/Webscrapping/postgraduate_master_degrees/maininfo",
    ]   
    _use_process_textsfiles_with_metadata_for_multiple_folders(folder_paths)

# ==== Processing, Saving and Loading Functions for Raw Extracted Text Files ====
def _use_process_textsfiles_with_metadata_for_multiple_folders(folder_paths: List[str]):
    all_bachelors = {}
    all_postgrad_and_masters = {}

    for folder_path in folder_paths:
        folder_docs = _process_textsfiles_with_metadata(folder_path)

        for filename, content in folder_docs.items():
            degree = content["metadata"]["degree"]

            if degree == "bachelor":
                all_bachelors[filename] = content
            elif degree in {"postgraduate", "masters"}:
                all_postgrad_and_masters[filename] = content
            else:
                print(f"⚠️ Skipping unknown degree in file: {filename}")

    _save_dict_program_textfiles_to_pickle(all_bachelors, output_file_name="dict_bachelors_raw.pkl", 
                                         output_folder=dc.HP_PATH_RAW_EXTRACTED_DOCS_DICTS)
    _save_dict_program_textfiles_to_pickle(all_postgrad_and_masters, output_file_name="dict_postgrad_and_masters_raw.pkl", 
                                          output_folder=dc.HP_PATH_RAW_EXTRACTED_DOCS_DICTS)

def _process_textsfiles_with_metadata(folder_path: str) -> Dict[str, Dict]:
    docs_input = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        filename_lower = filename.lower()

        # Infer degree from filename
        if "postgraduate" in filename_lower:
            degree = "postgraduate"
        elif "master" in filename_lower:
            degree = "masters"
        elif "bachelor" in filename_lower:
            degree = "bachelor"
        else:
            degree = "unknown"

        # Infer doc_type from filename
        if "teachingstaff" in filename_lower or "teaching-staff" in filename_lower or "faculty" in filename_lower:
            doc_type = "teaching_staff"
        elif "study plan" in filename_lower or "study_plan" in filename_lower or "studyplan" in filename_lower:
            doc_type = "study_plan"
        elif "maininfo" in filename_lower or "main_info" in filename_lower or "main_course" in filename_lower:
            doc_type = "main_info"
        else:
            doc_type = "unknown"

        # Extract course name from filename
        # Remove leading "bachelor_", "master_", etc., and trailing doc_type keywords
        course_part = filename_lower.replace(".txt", "")

        for prefix in ["bachelor_", "postgraduate_", "master_"]:
            if course_part.startswith(prefix):
                course_part = course_part[len(prefix):]

        for suffix in ["_teachingstaff", "_teaching-staff", "_faculty", "_study_plan", "_studyplan", "_study plan", "_maininfo", "_main_info", "_main_course", "_text"]:
            course_part = course_part.replace(suffix, "")

        course_name = course_part.replace("-", " ").replace("_", " ").strip().title()

        docs_input[filename] = {
            "text": text,
            "metadata": {
                "degree": degree,
                "doc_type": doc_type,
                "course_name": course_name
            }
        }

    return docs_input



