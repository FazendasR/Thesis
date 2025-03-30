import re

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


