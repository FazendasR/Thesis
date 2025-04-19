import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.figure_factory as ff
import tiktoken
from nltk.util import ngrams

nltk.download("punkt")

def visualize_general_text_statistics(text_data):
    """
    Generates visualizations for word count, sentence count, and average words per sentence 
    for a given text dataset.

    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are text content.

    Outputs:
    - Three bar charts visualizing:
      1. Word count per document
      2. Sentence count per document
      3. Average words per sentence per document
    """
    stats = {
        "filename": [],
        "word_count": [],
        "sentence_count": [],
        "avg_words_per_sentence": []
    }

    for filename, content in text_data.items():
        words = word_tokenize(content)
        sentences = sent_tokenize(content)

        stats["filename"].append(filename)
        stats["word_count"].append(len(words))
        stats["sentence_count"].append(len(sentences))
        stats["avg_words_per_sentence"].append(len(words) / len(sentences) if sentences else 0)

    # Set up the plot style
    sns.set_style("whitegrid")

    # Define a general function for plotting
    def plot_stat(x, y, title, ylabel, color_palette):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=x, y=y, palette=color_palette)
        plt.xticks(rotation=45, ha="right")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Filename", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.show()

    # Generate visualizations
    plot_stat(stats["filename"], stats["word_count"], "Word Count per Document", "Word Count", "Blues_r")
    plot_stat(stats["filename"], stats["sentence_count"], "Sentence Count per Document", "Sentence Count", "Greens_r")
    plot_stat(stats["filename"], stats["avg_words_per_sentence"], "Average Words per Sentence", "Avg Words per Sentence", "Oranges_r")


def visualize_word_frequencies(text_data, selected_files=None, top_n=10):
    """
    Visualizes word frequency distributions for selected text files using Seaborn.

    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are text content.
    - selected_files (list, optional): List of filenames to analyze. If None, all files are used.
    - top_n (int, optional): Number of top words to visualize (default: 10).

    Outputs:
    - Seaborn bar plots showing the top `top_n` most frequent words in each selected file.
    """
    if selected_files is None:
        selected_files = text_data.keys()

    for filename in selected_files:
        if filename not in text_data:
            print(f"Warning: {filename} not found in dataset. Skipping.")
            continue
        
        content = text_data[filename]
        words = word_tokenize(content)
        words = [word.lower() for word in words if word.isalnum()]  # Remove punctuation
        
        word_freq = Counter(words)
        common_words = word_freq.most_common(top_n)

        # Extract words and frequencies
        if not common_words:
            print(f"Warning: No words found in {filename}. Skipping visualization.")
            continue

        words, frequencies = zip(*common_words)

        # Seaborn Barplot
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")  # Set seaborn style
        ax = sns.barplot(x=list(words), y=list(frequencies), palette="Blues_r")

        # Formatting
        ax.set_title(f"Top {top_n} Most Frequent Words in {filename}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Words", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xticklabels(words, rotation=45)

        plt.show()

def visualize_keywords(text_data, selected_files=None, top_n=10):
    """
    Extracts and visualizes the top `top_n` keywords from selected text files using TF-IDF.

    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are text content.
    - selected_files (list, optional): List of filenames to analyze. If None, all files are used.
    - top_n (int, optional): Number of top keywords to extract (default: 10).

    Outputs:
    - Print and visualizations of the top `top_n` keywords for each selected file.
    """
    if selected_files is None:
        selected_files = text_data.keys()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)

    for filename in selected_files:
        if filename not in text_data:
            print(f"Warning: {filename} not found in dataset. Skipping.")
            continue
        
        content = text_data[filename]
        tfidf_matrix = vectorizer.fit_transform([content])
        keywords = vectorizer.get_feature_names_out()

        # Print top keywords
        print(f"--- Keywords in {filename} ---")
        print(", ".join(keywords))
        print("="*50)

        # Visualize the top keywords
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=list(keywords), y=tfidf_matrix.sum(axis=0).A1, palette="Greens_r")
        
        # Formatting the plot
        ax.set_title(f"Top {top_n} Keywords in {filename}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Keywords", fontsize=12)
        ax.set_ylabel("TF-IDF Score", fontsize=12)
        ax.set_xticklabels(keywords, rotation=45)
        
        plt.show()


import plotly.express as px
from typing import Dict, Optional, List

def histogram_word_count_multiple_docs(
    data: Dict[str, Dict],
    bins: int = 10,
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None
):
    """
    Plots a histogram of word counts from a dict of documents with metadata.

    Parameters:
    - data (dict): Dictionary with filenames as keys and dicts containing 'text' and 'metadata'.
    - bins (int): Number of bins for the histogram.
    - course_names_to_include (list, optional): Filter by specific course names (case insensitive).
    - doc_types_to_include (list, optional): Filter by document types ('teaching_staff', 'study_plan', 'main_info').
    """

    filtered_data = {}

    for filename, doc in data.items():
        course_name = doc["metadata"].get("course_name", "").lower()
        doc_type = doc["metadata"].get("doc_type", "").lower()

        if course_names_to_include:
            if course_name not in [name.lower() for name in course_names_to_include]:
                continue

        if doc_types_to_include:
            if doc_type not in [dt.lower() for dt in doc_types_to_include]:
                continue

        filtered_data[filename] = doc

    if not filtered_data:
        print("⚠️ No documents matched the filters.")
        return

    word_counts = [len(doc["text"].split()) for doc in filtered_data.values()]
    labels = [f"{doc['metadata'].get('course_name')} ({doc['metadata'].get('doc_type')})" for doc in filtered_data.values()]

    fig = px.histogram(
        x=word_counts,
        nbins=bins,
        labels={'x': 'Word Count', 'y': 'Number of Documents'},
        title='Distribution of Documents by Word Count'
    )

    fig.update_layout(
        xaxis_title="Word Count",
        yaxis_title="Number of Documents",
        bargap=0.1
    )

    fig.show()
from typing import Dict, Optional, List
import plotly.express as px
import tiktoken

def histogram_token_count_multiple_docs(
    data: Dict[str, Dict],
    model: str = "gpt-4",
    bins: int = 15,
    course_names_to_include: Optional[List[str]] = None,
    doc_types_to_include: Optional[List[str]] = None,
):
    """
    Plots a histogram of token counts per document using Plotly with optional filters.
    
    Parameters:
    - data (dict): Dictionary with filenames as keys and values as dicts with 'text' and 'metadata'.
    - model (str): OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo").
    - bins (int): Number of bins for the histogram.
    - course_names_to_include (list): Optional list of course names to include.
    - doc_types_to_include (list): Optional list of document types to include.
    """
    # Load tokenizer
    encoding = tiktoken.encoding_for_model(model)

    # Apply filters
    filtered_items = []
    for filename, doc in data.items():
        metadata = doc.get("metadata", {})
        course_name = metadata.get("course_name", "")
        doc_type = metadata.get("doc_type", "")

        if course_names_to_include and course_name not in course_names_to_include:
            continue
        if doc_types_to_include and doc_type not in doc_types_to_include:
            continue

        filtered_items.append((filename, doc["text"]))

    if not filtered_items:
        print("⚠️ No documents matched the filters.")
        return

    # Compute token counts
    token_counts = {
        filename: len(encoding.encode(text))
        for filename, text in filtered_items
    }

    # Plot
    fig = px.histogram(
        x=list(token_counts.values()),
        nbins=bins,
        labels={'x': 'Token Count', 'y': 'Number of Documents'},
        title='Distribution of Documents by Token Count'
    )

    fig.update_layout(
        xaxis_title="Token Count",
        yaxis_title="Number of Documents",
        bargap=0.1
    )

    fig.show()

def histogram_token_count_multiple_docs_old(text_data, model="gpt-4", bins=15):
    """
    Plots a histogram of token counts per document using Plotly.
    
    Parameters:
    - text_data (dict): Dictionary with filenames as keys and text content as values.
    - model (str): OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo").
    - bins (int): Number of bins for the histogram.
    - max_tokens (int): Maximum token limit to check against.
    """
    
    # Load the appropriate tokenizer
    encoding = tiktoken.encoding_for_model(model)
    
    # Compute token count for each document
    token_counts = {filename: len(encoding.encode(content)) for filename, content in text_data.items()}
    
    # Convert to list for histogram
    token_values = list(token_counts.values())

    # Create histogram
    fig = px.histogram(
        x=token_values,
        nbins=bins,
        labels={'x': 'Token Count', 'y': 'Number of Documents'},
        title=f'Distribution of Documents by Token Count'
    )

    fig.update_layout(
        xaxis_title="Token Count",
        yaxis_title="Number of Documents",
        bargap=0.1
    )
    
    # Show figure
    fig.show()

def generate_document_statistics_by_word_count(text_data):
    """
    Generate and visualize document statistics for a given text dataset.
    
    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    
    Returns:
    - Plotly table with summary statistics for word counts in the documents.
    """
    # Calculate word counts for each document
    word_counts = [len(word_tokenize(content)) for content in text_data.values()]
    
    # Compute summary statistics
    summary = pd.DataFrame(word_counts, columns=["Word Count"]).describe().T

    # Reshape the data to vertical format
    summary = summary.T.reset_index()
    summary.columns = ["Metric", "Value"]

    # Format numeric values for better readability
    summary["Value"] = summary["Value"].apply(lambda x: f"{x:,.2f}")

    # Create interactive Plotly table
    fig = ff.create_table(summary, index=False)

    # Center-align the table content
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(align="center")

    fig.update_layout(title="Document Statistics")
    
    fig.show()

def generate_statistics_for_all_documents(dict_of_dicts, model="gpt-4"):
    """
    Applies `generate_document_statistics_by_tokens` to each sub-dictionary in a dict of dicts.

    :param dict_of_dicts: A dictionary where each value is another dictionary of documents.
    :param model: The OpenAI model name to use for tokenization.
    """
    for category_name, sub_dict in dict_of_dicts.items():
        print(f"\nGenerating stats for: {category_name}")
        generate_document_statistics_by_tokens(sub_dict, model=model)

def generate_document_statistics_by_tokens(text_data, model="gpt-4"):
    """
    Generate and visualize document statistics for token counts in a given text dataset.
    
    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    - model (str): OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo").
    
    Returns:
    - Plotly table with summary statistics for token counts in the documents.
    """
    # Load the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    
    # Calculate token counts for each document
    token_counts = [len(encoding.encode(content)) for content in text_data.values()]
    
    # Compute summary statistics
    summary = pd.DataFrame(token_counts, columns=["Token Count"]).describe().T

    # Reshape the data to vertical format
    summary = summary.T.reset_index()
    summary.columns = ["Metric", "Value"]

    # Format numeric values for better readability
    summary["Value"] = summary["Value"].apply(lambda x: f"{x:,.2f}")

    # Create interactive Plotly table
    fig = ff.create_table(summary, index=False)

    # Center-align the table content
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(align="center")

    fig.update_layout(title="Token-Based Document Statistics")

    fig.show()


def bar_plot_word_frequency(text_data, top_n=20):
    """
    Generates a Plotly bar chart of the most frequent words in a given text dataset.
    
    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    - top_n (int): Number of top words to display.
    
    Returns:
    - A Plotly bar plot showing word frequency with counts and percentages.
    """
    
    # Tokenize and normalize words (remove punctuation, convert to lowercase)
    words = []
    for content in text_data.values():
        tokens = word_tokenize(content)
        words.extend([word.lower() for word in tokens if word.isalnum()])  # Keep only alphanumeric words
    
    # Compute word frequency
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(top_n)

    # Convert to DataFrame for visualization
    df = pd.DataFrame(most_common_words, columns=["Word", "Count"])
    
    # Calculate percentage
    total_words = sum(word_freq.values())
    df["Percentage"] = (df["Count"] / total_words) * 100

    # Create Plotly bar chart
    fig = px.bar(
        df, 
        x="Word", 
        y="Count",
        text=df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.2f}%)", axis=1),
        title=f"Top {top_n} Most Frequent Words",
        labels={"Word": "Word", "Count": "Frequency"},
        color="Count",
        color_continuous_scale="Blues"
    )

    # Update layout
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Frequency",
        coloraxis_showscale=False,
        bargap=0.2
    )

    # Show figure
    fig.show()

def bar_plot_ngram_frequency(text_data, n=2, top_n=20):
    """
    Generates a Plotly bar chart of the most frequent n-grams in a given text dataset.
    
    Parameters:
    - text_data (dict): Dictionary where keys are filenames and values are document contents.
    - n (int): The 'n' in n-gram (default is 2 for bigrams).
    - top_n (int): Number of top n-grams to display.
    
    Returns:
    - A Plotly bar plot showing n-gram frequency with counts and percentages.
    """
    
    # Tokenize and normalize words (remove punctuation, convert to lowercase)
    all_ngrams = []
    for content in text_data.values():
        tokens = word_tokenize(content)
        words = [word.lower() for word in tokens if word.isalnum()]  # Keep only alphanumeric words
        n_grams = list(ngrams(words, n))
        all_ngrams.extend(n_grams)
    
    # Compute n-gram frequency
    ngram_freq = Counter(all_ngrams)
    most_common_ngrams = ngram_freq.most_common(top_n)

    # Convert to DataFrame for visualization
    df = pd.DataFrame(most_common_ngrams, columns=["N-Gram", "Count"])
    
    # Convert tuples to readable strings
    df["N-Gram"] = df["N-Gram"].apply(lambda x: " ".join(x))
    
    # Calculate percentage
    total_ngrams = sum(ngram_freq.values())
    df["Percentage"] = (df["Count"] / total_ngrams) * 100

    # Create Plotly bar chart
    fig = px.bar(
        df, 
        x="N-Gram", 
        y="Count",
        text=df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.2f}%)", axis=1),
        title=f"Top {top_n} Most Frequent {n}-Grams",
        labels={"N-Gram": "N-Gram", "Count": "Frequency"},
        color="Count",
        color_continuous_scale="Blues"
    )

    # Update layout
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title=f"{n}-Grams",
        yaxis_title="Frequency",
        coloraxis_showscale=False,
        bargap=0.2
    )

    # Show figure
    fig.show()
