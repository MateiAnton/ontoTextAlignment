import argparse
import pandas as pd
import nltk
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from tqdm import tqdm
import re
import networkx as nx

# Setup progress bar support with pandas
tqdm.pandas()

def remove_non_english_characters(text):
    # Remove characters not typically found in English language texts
    pattern = '[^a-zA-Z0-9\s.,!?;:]'
    return re.sub(pattern, '', text)

def preprocess(text):
    # Clean and tokenize text, removing stopwords
    cleaned_text = remove_non_english_characters(text).lower()
    word_tokens = word_tokenize(cleaned_text)
    return " ".join(word for word in word_tokens if word not in stopwords.words('english'))

def rank_sentences_with_vectorizer(vectorizer, row, sentences, sentence_ids):
    # Calculate cosine similarity and rank sentences
    start_time = time.time()
    target_text = preprocess(row.get('Annotation Text', row.get('Query')))
    vectors = vectorizer.fit_transform([target_text] + sentences)
    cosine_scores = cosine_similarity(vectors[0:1], vectors[1:])
    ranked_sentence_ids = [sentence_ids[i] for i in np.argsort(cosine_scores[0])[::-1]]
    return ranked_sentence_ids, time.time() - start_time

def main(data_file, json_file):
    # Setup nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    # Read the dataset based on the file extension
    file_extension = os.path.splitext(data_file)[1].lower()
    if file_extension == '.pkl':
        df = pd.read_pickle(data_file)
    elif file_extension == '.csv':
        df = pd.read_csv(data_file)
    elif file_extension == '.json':
        df = pd.read_json(data_file)
    else:
        raise ValueError("Unsupported file type. Please provide a .pkl, .csv, or .json file.")

    # Load sentences from JSON
    with open(json_file, 'r') as file:
        sentence_dict = json.load(file)
    sentences = [preprocess(sentence) for sentence in sentence_dict.values()]
    sentence_ids = list(sentence_dict.keys())

    # Setup vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Rank sentences
    results = df.progress_apply(lambda row: rank_sentences_with_vectorizer(tfidf_vectorizer, row, sentences, sentence_ids), axis=1)
    df['TfidfVec Cosine'] = results.apply(lambda x: x[0])
    df['TfidfVec Time'] = results.apply(lambda x: x[1])

    # Save the DataFrame
    output_filename = os.path.splitext(data_file)[0] + "_TextRank.pkl"
    df.to_pickle(output_filename)
    print(f"DataFrame saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .pkl, .csv, or .json files to compute cosine similarities.")
    parser.add_argument("data_file", type=str, help="Path to the data file (.pkl, .csv, .json)")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing sentence data")
    args = parser.parse_args()

    main(args.data_file, args.json_file)
