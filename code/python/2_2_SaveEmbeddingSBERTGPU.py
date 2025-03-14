import sys
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import tensorflow as tf

def main(argv):
    """
    This script processes a JSON file containing ontology axioms to calculate and save their embeddings.
    
    Args:
    argv (list): Command line arguments where argv[1] is the JSON file path.
    """
    if len(argv) != 2:
        print(f'Usage: {argv[0]} <path_to_ontology_json_file>')
        sys.exit(1)

    ontology_file = argv[1]
    
    # Load JSON data containing axiom sentences
    try:
        with open(ontology_file, 'r') as json_file:
            id_axiom_sentence_dict = json.load(json_file)
        print(f"Loaded {len(id_axiom_sentence_dict)} axioms.")
    except FileNotFoundError:
        print(f"Error: File '{ontology_file}' not found.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON data. It might be invalid JSON.")
        return

    # Initialize the SentenceTransformer model on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(device)
    
    # Function to get BERT embedding for a given text
    def get_bert_embedding(text):
        return model.encode(text, convert_to_tensor=True).to(device)
    
    # Compute embeddings for each sentence in the dictionary
    sentence_embeddings = {key: get_bert_embedding(sentence) for key, sentence in tqdm(id_axiom_sentence_dict.items())}
    
    # Save embeddings to a file
    file_path = f"{ontology_file}_SBERT_embeddings.pkl"
    sentence_embeddings_converted = {key: value.cpu().numpy() for key, value in sentence_embeddings.items()}
    
    # Use TensorFlow's file I/O to save the numpy arrays
    with tf.io.gfile.GFile(file_path, "wb") as file:
        np.save(file, sentence_embeddings_converted)
    print(f"Embeddings saved to '{file_path}'.")

if __name__ == "__main__":
    main(sys.argv)

