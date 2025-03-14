import sys
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np

def get_bert_embedding(text, tokenizer, model, device):
    """Tokenize and convert text to BERT embeddings."""
    # Tokenize the input text, ensuring padding and truncation, converting to PyTorch tensors.
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def main(argv):
    """
    This script loads JSON data with ontology axioms, computes their BERT embeddings, and saves these embeddings.

    Args:
    argv (list): Command line arguments where argv[1] is the JSON file path.
    """
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <path_to_ontology_json_file>")
        sys.exit(1)

    ontology_file = argv[1]

    # Load JSON data containing ontology axioms
    try:
        with open(ontology_file, 'r') as json_file:
            id_axiom_sentence_dict = json.load(json_file)
        print(f"Loaded {len(id_axiom_sentence_dict)} axioms.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse the file '{ontology_file}': {str(e)}")
        return

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

#    # Embed input text
#    input_text = "The symptoms of COVID‑19 are variable but often include fever, cough, headache, fatigue, breathing difficulties, loss of smell, and loss of taste."
#    input_embedding = get_bert_embedding(input_text, tokenizer, model, device)
#
    # Compute embeddings for each sentence in the ontology
    sentence_embeddings = {key: get_bert_embedding(sentence, tokenizer, model, device) for key, sentence in tqdm(id_axiom_sentence_dict.items())}
#
#    # Calculate cosine similarities between input text and each ontology sentence
#    similarities = {key: cosine_similarity(input_embedding.cpu().numpy(), sentence.cpu().numpy())[0][0] for key, sentence in sentence_embeddings.items()}

    # Save embeddings to a file
    file_path = ontology_file + "_SapBERT_embeddings.pkl"
    sentence_embeddings_converted = {key: value.detach().cpu().numpy() for key, value in sentence_embeddings.items()}

    # Using TensorFlow's file I/O to save the numpy arrays
    with tf.io.gfile.GFile(file_path, "wb") as file:
        np.save(file, sentence_embeddings_converted)
    print(f"Embeddings saved to '{file_path}'.")

if __name__ == "__main__":
    main(sys.argv)
