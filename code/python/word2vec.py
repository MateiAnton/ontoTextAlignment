import argparse
import pandas as pd
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import time
from tqdm import tqdm
import re
import gensim.downloader as api
from scipy.spatial.distance import cosine


# Necessary for tqdm progress bar support with pandas
tqdm.pandas()

def remove_non_english_characters(text):
    # This pattern matches any character that is not a basic English letter (both cases), number, or common punctuation
    pattern = '[^a-zA-Z0-9\s.,!?;:]'
    return re.sub(pattern, '', text)

# Function to preprocess text
def preprocess(text):
    cleaned_text = remove_non_english_characters(text)
    lower_text = cleaned_text.lower()
    word_tokens = word_tokenize(lower_text)
    filtered_words = [word for word in word_tokens if word not in stopwords.words('english')]
    return " ".join(filtered_words)


# Load Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')

def sentence_to_avg_vector(sentence, model):
    words = word_tokenize(sentence.lower())
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(word_vectors, axis=0)

def rank_sentences_with_word2vec(row, sentences, sentence_ids, model):
    start_time = time.time()
    # target_vector = sentence_to_avg_vector(preprocess(row['Annotation Text']), model)
    # Choose the column 'Annotation Text' if it exists, otherwise use 'Query'
    column_name = 'Annotation Text' if 'Annotation Text' in row else 'Query'
    target_vector = sentence_to_avg_vector(preprocess(row[column_name]), model)

    sentence_vectors = [sentence_to_avg_vector(sentence, model) for sentence in sentences]

    cosine_scores = [1 - cosine(target_vector, vec) for vec in sentence_vectors]
    ranked_sentence_ids = [sentence_ids[i] for i in np.argsort(cosine_scores)[::-1]]
    end_time = time.time()
    computation_time = end_time - start_time
    return ranked_sentence_ids, computation_time

def main(data_file, json_file):
    nltk.download('punkt')
    nltk.download('stopwords')


    # Determine the file extension and read the file accordingly
    file_extension = os.path.splitext(data_file)[1].lower()
    if file_extension == '.pkl':
        df = pd.read_pickle(data_file)
    elif file_extension == '.csv':
        df = pd.read_csv(data_file,delimiter=';')
    elif file_extension == '.json':
        df = pd.read_json(data_file)
    else:
        raise ValueError("Unsupported file type. Please provide a .pkl, .csv, or .json file.")


    with open(json_file, 'r') as file:
        sentence_dict = json.load(file)

    sentences = [preprocess(sentence) for sentence in sentence_dict.values()]
    sentence_ids = list(sentence_dict.keys())

    # Apply the function with Word2Vec
    word2vec_results = df.progress_apply(lambda row: rank_sentences_with_word2vec(row, sentences, sentence_ids, word2vec_model), axis=1)
    df['Word2Vec Cosine'] = word2vec_results.apply(lambda x: x[0])
    df['Word2Vec Time'] = word2vec_results.apply(lambda x: x[1])

    # Save the DataFrame
    output_filename = os.path.splitext(data_file)[0] + "_Word2VecCos.pkl"
    df.to_pickle(output_filename)
    print(f"DataFrame saved to {output_filename}")
    print(df)

# The rest of your script including the if __name__ == "__main__": block should also be properly indented
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .pkl or .csv and .json files to compute cosine similarities.")
    parser.add_argument("data_file", type=str, help="Path to the .pkl or .csv file")
    parser.add_argument("json_file", type=str, help="Path to the .json file")
    args = parser.parse_args()

    main(args.data_file, args.json_file)