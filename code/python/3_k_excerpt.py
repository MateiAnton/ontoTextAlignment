import numpy as np
import pandas as pd
import torch
# from tqdm.auto import tqdm
# from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# import tensorflow as tf
import faiss
from flashtext import KeywordProcessor
from gensim.models import KeyedVectors
from nltk.tokenize import MWETokenizer

# # file_path = "/home/jieying/ontologies/snomed2021_id_to_axiom_sentence.jsonberg_large_uncased_dict_file.pkl"
# # file_path = "/home/jieying/ontologies/snomed2021_id_to_axiom_sentence.jsonSBERT_dict_file.pkl"
file_path = "generated_data/Merged_GeoFault_modified.ttl_id_to_axiom_sentence"
file_path_BERT =file_path + ".json_BERT_embeddings.pkl"
file_path_SBERT =file_path + ".json_SBERT_embeddings.pkl"
file_path_SapBERT =file_path + ".json_SapBERT_embeddings.pkl"
file_path_owl2vec_iri =file_path + "_iri.json_owl2vec_embeddings.pkl"
file_path_owl2vec =file_path + ".json_owl2vec_embeddings.pkl"
file_path_owl2vec_pretrained =file_path + ".json_owl2vec_pretrained_embeddings.pkl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CUDA_VISIBLE_DEVICE=0

# with open(file_path, "rb") as file:
#     berg_axiom_vectors = np.load(file, allow_pickle=True).item()

        # print(loaded_dict_converted)
# Convert numpy arrays back to tensors if necessary and move to CPU

# BERT_larget_embedding
def get_bert_embedding(text):
    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    # tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**tokens)
    #     embeddings = torch.mean(outputs[0], dim=1)
    # return embeddings
    # tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    tokens = tokenizer(text, max_length=512,padding=True, truncation=True, return_tensors="pt")
     # Move tokens to the GPU
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = torch.mean(outputs[0], dim=1)
    return embeddings



def k_excerpt_BERT(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            berg_axiom_vectors = np.load(file, allow_pickle=True).item()
    else:
        berg_axiom_vectors = arg2
    
    # Convert dictionary values to a list of numpy arrays and reshape them to 1D
    berg_axiom_vector_list = [sentence.squeeze() for sentence in berg_axiom_vectors.values()]
    
    # Creating the array from the list
    berg_axiom_vector_array = np.array(berg_axiom_vector_list).astype("float32")

    # Check and print the shape of the final array
    # print(f"Final array shape: {berg_axiom_vector_array.shape}")

    # Ensure that berg_axiom_vector_array is 2D: (num_vectors, vector_dimension)
    if berg_axiom_vector_array.ndim != 2:
        raise ValueError("Embedding array is not 2-dimensional")

    # Creating a FAISS index
    dimension = berg_axiom_vector_array.shape[1]  # assuming all vectors have the same dimension
    faiss_index = faiss.IndexFlatL2(dimension)  # using L2 distance for similarity
    faiss_index.add(berg_axiom_vector_array)  # adding the vectors to the index

    # Embed the input text
    input_embedding = get_bert_embedding(input_text).cpu().numpy().reshape(1, -1)

   # Determining the value of k
    if k is None:
        A = len(berg_axiom_vectors) / 100
        k = 100 if A < 100 else int(A)

    

    if k > len(berg_axiom_vectors):
        k = len(berg_axiom_vectors)

    _, indices = faiss_index.search(input_embedding, k)

    # Compute cosine similarities for the top k results
    similarities = {}
    for idx in indices[0]:
        key = list(berg_axiom_vectors.keys())[idx]
        sentence_embedding = berg_axiom_vector_array[idx].reshape(1, -1)
        similarity_score = cosine_similarity(input_embedding, sentence_embedding)[0][0]
        similarities[key] = similarity_score

    # Sorting by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_axiom_sentences = dict(sorted_similarities)
    # print(sorted_similarities)

    return top_k_axiom_sentences


# WITHOUT FAISS INDEXING
def k_excerpt_BERT_withoutFAISS(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            berg_axiom_vectors = np.load(file, allow_pickle=True).item()
    else:
        berg_axiom_vectors = arg2

    input_embedding = get_bert_embedding(input_text)
    similarities = {key: cosine_similarity(input_embedding.cpu().numpy(), sentence.cpu().numpy())[0][0] for key, sentence in berg_axiom_vectors.items()}

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    if k is not None:
        top_k_axiom_sentences = dict(sorted_similarities[:k])
    else:
        top_k_axiom_sentences = dict(sorted_similarities)

    return top_k_axiom_sentences

# def k_excerpt_BERT (input_text,berg_axiom_vectors,k):
#     input_embedding = get_bert_embedding(input_text)
#     # sentence_embeddings = {key: get_bert_embedding(sentence) for key, sentence in tqdm(id_axiom_sentence_dict.items())}
#     similarities = {key: cosine_similarity(input_embedding.cpu().numpy(), sentence.cpu().numpy())[0][0] for key, sentence in tqdm(berg_axiom_vectors.items())}

#     top_k_axiom_sentences = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k])
#     # for key, score in top_k_axiom_sentences.items():
#     #     # excerpt_text =""
#     #     # excerpt_text = excerpt_text + berg_axiom_vectors[key]
#     #     print(f"ID: {key}, Sentence: {berg_axiom_vectors[key]}, Similarity Score: {score}")
#     return top_k_axiom_sentences



# Setence BERT embedding
def get_SBERT_embedding(text):
    # Initialize the model
    # model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    # # tf.debugging.set_log_device_placement(True)
    # return model.encode(text, convert_to_tensor=True).to(device)
    if not isinstance(text, (str, list)):
        raise ValueError(f"Expected text to be a string or list of strings, but got {type(text)} with value {text}")

    # Initialize the model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    return model.encode(text, convert_to_tensor=True).to(device)

def k_excerpt_SBERT_withoutFAISS(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            id_axiom_sentence_dict = np.load(file, allow_pickle=True).item()
    else:
        id_axiom_sentence_dict = arg2

    input_embedding = get_SBERT_embedding(input_text)
    similarities = {key: torch.nn.functional.cosine_similarity(input_embedding.unsqueeze(0), sentence.unsqueeze(0)).cpu().numpy()[0] for key, sentence in id_axiom_sentence_dict.items()}

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    if k is not None:
        top_k_sentences = dict(sorted_similarities[:k])
    else:
        top_k_sentences = dict(sorted_similarities)

    return top_k_sentences

###################### after semantic indexing ###################### 

def k_excerpt_SBERT(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            id_axiom_sentence_dict = np.load(file, allow_pickle=True).item()
    else:
        id_axiom_sentence_dict = arg2

    # Convert dictionary values to a list of numpy arrays
    id_axiom_vector_list = [sentence for sentence in id_axiom_sentence_dict.values()]

    # Creating the array from the list
    id_axiom_vector_array = np.array(id_axiom_vector_list).astype("float32")

    # Ensure that id_axiom_vector_array is 2D: (num_vectors, vector_dimension)
    if id_axiom_vector_array.ndim != 2:
        raise ValueError("Embedding array is not 2-dimensional")

    # Creating a FAISS index
    dimension = id_axiom_vector_array.shape[1]  # assuming all vectors have the same dimension
    faiss_index = faiss.IndexFlatL2(dimension)  # using L2 distance for similarity
    faiss_index.add(id_axiom_vector_array)  # adding the vectors to the index

    # Embed the input text with SBERT
    input_embedding = get_SBERT_embedding(input_text).cpu().numpy()

    # Ensure input_embedding is 2D for FAISS
    if input_embedding.ndim == 1:
        input_embedding = input_embedding.reshape(1, -1)

    # Determining the value of k
    if k is None:
        A = len(id_axiom_sentence_dict) / 100
        k = 100 if A < 100 else int(A)

    if k > len(id_axiom_sentence_dict):
        k = len(id_axiom_sentence_dict)

    

    _, indices = faiss_index.search(input_embedding, k)

    # Compute cosine similarities for the top k results
    similarities = {}
    for idx in indices[0]:
        key = list(id_axiom_sentence_dict.keys())[idx]
        sentence_embedding = id_axiom_vector_array[idx].reshape(1, -1)
        similarity_score = cosine_similarity(input_embedding, sentence_embedding)[0][0]
        similarities[key] = similarity_score

    # Sorting by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_sentences = dict(sorted_similarities)

    return top_k_sentences



# SapBERT
def get_SapBERT_embedding(text):
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    tokens = tokenizer(text,max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = torch.mean(outputs[0], dim=1)
    return embeddings

def k_excerpt_SapBERT_withoutFAISS(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            id_axiom_sentence_dict = np.load(file, allow_pickle=True).item()
    else:
        id_axiom_sentence_dict = arg2

    input_embedding = get_SapBERT_embedding(input_text)

    similarities = {}
    for key, sentence in id_axiom_sentence_dict.items():
        if input_embedding.shape[1] == sentence.shape[1]:
            similarity = cosine_similarity(input_embedding.cpu().numpy(), sentence.cpu().numpy())[0][0]
            similarities[key] = similarity
        else:
            print(f"Dimension mismatch for key {key}: input {input_embedding.shape}, sentence {sentence.shape}")

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    if k is not None:
        top_k_sentences = dict(sorted_similarities[:k])
    else:
        top_k_sentences = dict(sorted_similarities)

    return top_k_sentences

# Update after using FAISS indexing
def k_excerpt_SapBERT(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            id_axiom_sentence_dict = np.load(file, allow_pickle=True).item()
    else:
        id_axiom_sentence_dict = arg2

    # Convert dictionary values to a list of numpy arrays and reshape them to 1D
    axiom_vector_list = [sentence.squeeze() for sentence in id_axiom_sentence_dict.values()]
    axiom_vector_array = np.array(axiom_vector_list).astype("float32")

    # Ensure that axiom_vector_array is 2D
    if axiom_vector_array.ndim != 2:
        raise ValueError("Embedding array is not 2-dimensional")

    # Creating a FAISS index
    dimension = axiom_vector_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(axiom_vector_array)

    # Embed the input text
    input_embedding = get_SapBERT_embedding(input_text).cpu().numpy().reshape(1, -1)

    # Determining the value of k
    if k is None:
        A = len(id_axiom_sentence_dict) / 100
        k = 100 if A < 100 else int(A)

    if k > len(id_axiom_sentence_dict):
        k = len(id_axiom_sentence_dict)

    

    _, indices = faiss_index.search(input_embedding, k)

    # Compute cosine similarities for the top k results
    similarities = {}
    for idx in indices[0]:
        key = list(id_axiom_sentence_dict.keys())[idx]
        sentence_embedding = axiom_vector_array[idx].reshape(1, -1)
        similarity_score = cosine_similarity(input_embedding, sentence_embedding)[0][0]
        similarities[key] = similarity_score

    # Sorting by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_sentences = dict(sorted_similarities)
    # print(top_k_sentences)

    return top_k_sentences

def get_owl2vec_embedding(text):
    # Load the owl2vec model
    model = KeyedVectors.load("/var/scratch/man471/cache/output_pretrained/ontology.embeddings", mmap='r')
    wv = model.wv
    keys = wv.index_to_key

    text = text.replace(".", "").replace(",", "").replace(":", "").replace(";", "").lower()
    text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "").replace("`", "")
    text = text.replace("!", "").replace("?", "")
    text = text.replace("*", "").replace("+", "").replace("=", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("|", "").replace("\\", "").replace("/", "").replace("~", "")
    text = text.replace("`", "").replace("´", "").replace("`", "").replace("‘", "")
    text = text.replace("“", "").replace("”", "").replace("'", "").replace("’", "")

    
    tokenizer = MWETokenizer()

    keyword_processor = KeywordProcessor()

    keyword_processor.add_keywords_from_list(keys)

    keywords_found = keyword_processor.extract_keywords(text, span_info=True)

    for a in keywords_found:
        tokenizer.add_mwe(text[a[1]: a[2]].split())

    tokens = tokenizer.tokenize(text.split())

    embeddings = []
    for token in tokens:
        if token in wv.key_to_index:
            embeddings.append(wv[token])
        elif token.replace("_", " ") in wv.key_to_index:
            embeddings.append(wv[token.replace("_", " ")])
        elif token.replace("-", " ") in wv.key_to_index:
            embeddings.append(wv[token.replace("-", " ")])
        elif token[-1] == "s" and token[:-1] in wv.key_to_index:
            embeddings.append(wv[token[:-1]])
        else:
            # print(token.replace("_", ""))
            print(f"Token '{token}' not found in the model vocabulary.")

    return np.mean(embeddings, axis=0) if embeddings else np.zeros(wv.vector_size)

def k_excerpt_owl2vec(input_text, arg2, k=None, is_file_path=True):
    if is_file_path:
        with open(arg2, "rb") as file:
            id_axiom_sentence_dict = np.load(file, allow_pickle=True).item()
    else:
        id_axiom_sentence_dict = arg2

    # Convert dictionary values to a list of numpy arrays and reshape them to 1D
    axiom_vector_list = [sentence.squeeze() for sentence in id_axiom_sentence_dict.values()]
    axiom_vector_array = np.array(axiom_vector_list).astype("float32")

    # Ensure that axiom_vector_array is 2D
    if axiom_vector_array.ndim != 2:
        raise ValueError("Embedding array is not 2-dimensional")

    # Creating a FAISS index
    dimension = axiom_vector_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(axiom_vector_array)

    # Embed the input text
    input_embedding = get_owl2vec_embedding(input_text).reshape(1, -1)

    # Determining the value of k
    if k is None:
        A = len(id_axiom_sentence_dict) / 100
        k = 100 if A < 100 else int(A)

    if k > len(id_axiom_sentence_dict):
        k = len(id_axiom_sentence_dict)

    

    _, indices = faiss_index.search(input_embedding, k)

    # Compute cosine similarities for the top k results
    similarities = {}
    for idx in indices[0]:
        key = list(id_axiom_sentence_dict.keys())[idx]
        sentence_embedding = axiom_vector_array[idx].reshape(1, -1)
        similarity_score = cosine_similarity(input_embedding, sentence_embedding)[0][0]
        similarities[key] = similarity_score

    # Sorting by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_k_sentences = dict(sorted_similarities)
    
    return top_k_sentences


# input_text = "A fault is a planar fracture or discontinuity in a volume of rock across which there has been significant displacement as a result of rock-mass movements. Large faults within Earth's crust result from the action of plate tectonic forces, with the largest forming the boundaries between the plates, such as the megathrust faults of subduction zones or transform faults. Energy release associated with rapid movement on active faults is the cause of most earthquakes. Faults may also displace slowly, by aseismic creep."

# print(k_excerpt_BERT(input_text,file_path_BERT,5).keys())
# print(k_excerpt_SBERT(input_text,file_path_SBERT,5).keys())
# print(k_excerpt_SapBERT(input_text,file_path_SapBERT,5).keys())

benchmark_file_path = "./generated_data/new_GeoFaultBenchmark_with_rankings.csv"

# load benchmark file into pandas datastore with first row as header
benchmark_data = pd.read_csv(benchmark_file_path, header=0)
if 'BERT_Ranking' not in benchmark_data.columns:
    benchmark_data['BERT_Ranking'] = None
if 'SBERT_Ranking' not in benchmark_data.columns:
    benchmark_data['SBERT_Ranking'] = None
if 'SapBERT_Ranking' not in benchmark_data.columns:
    benchmark_data['SapBERT_Ranking'] = None
if 'owl2vec_iri_Ranking' not in benchmark_data.columns:
    benchmark_data['owl2vec_iri_Ranking'] = None
if 'owl2vec_Ranking' not in benchmark_data.columns:
    benchmark_data['owl2vec_Ranking'] = None
if 'owl2vec_pretrained_Ranking' not in benchmark_data.columns:
    benchmark_data['owl2vec_pretrained_Ranking'] = None

# benchmark_data['BERT_Ranking'] = benchmark_data.apply(
#     lambda row: k_excerpt_BERT(
#         row['Query'],
#         file_path_BERT, 10),
#         axis=1)
# benchmark_data['SBERT_Ranking'] = benchmark_data.apply(
#     lambda row: k_excerpt_SBERT(
#         row['Query'],
#         file_path_SBERT, 10),
#         axis=1)
# benchmark_data['SapBERT_Ranking'] = benchmark_data.apply(
#     lambda row: k_excerpt_SapBERT(
#         row['Query'],
#         file_path_SapBERT, 10),
#         axis=1)
# benchmark_data['owl2vec_Ranking'] = benchmark_data.apply(
#     lambda row: k_excerpt_owl2vec(
#         row['Query'],
#         file_path_owl2vec_iri, 10),
#         axis=1)
# benchmark_data['owl2vec_Ranking'] = benchmark_data.apply(
#     lambda row: k_excerpt_owl2vec(
#         row['Query'],
#         file_path_owl2vec, 10),
#         axis=1)
benchmark_data['owl2vec_pretrained_Ranking'] = benchmark_data.apply(
    lambda row: k_excerpt_owl2vec(
        row['Query'],
        file_path_owl2vec_pretrained, 10),
        axis=1)
benchmark_data.to_csv("./generated_data/new_GeoFaultBenchmark_with_rankings.csv", index=False)
