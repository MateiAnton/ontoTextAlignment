import json
import sys
import numpy as np
from gensim.models import KeyedVectors
import tensorflow as tf
from flashtext import KeywordProcessor
from gensim.models import KeyedVectors
from nltk.tokenize import MWETokenizer
from gensim.models import Word2Vec

def main(argv):
    """
    This script processes a JSON file containing ontology axioms to calculate and save their embeddings.
    
    Args:
    argv (list): Command line arguments where argv[1] is the JSON file path.
    """
    if len(argv) != 2:
        print(f'Usage: {argv[0]} <path_to_ontology_json_file>')
        sys.exit(1)

    ontology_json_file = argv[1]

    # Load JSON data containing axiom sentences
    try:
        with open(ontology_json_file, 'r') as json_file:
            id_axiom_sentence_dict = json.load(json_file)
        print(f"Loaded {len(id_axiom_sentence_dict)} axioms.")
    except FileNotFoundError:
        print(f"Error: File '{ontology_json_file}' not found.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON data. It might be invalid JSON.")
        return
    
    #Embedding vectors generated above
    model = KeyedVectors.load("/var/scratch/man471/cache/output_pretrained/ontology.embeddings", mmap='r')
    wv = model.wv
    # word2vec_model = Word2Vec.load("/home/matei/w2v_model/enwiki_model/word2vec_model")
    # Function to get Owl2Vec embedding for a given text
    def get_owl2vec_embedding(text):
        """
        Get the Owl2Vec embedding for a given text.
        
        Args:
        text (str): The input text to be embedded.
        
        Returns:
        np.ndarray: The embedding vector for the input text.
        """

        text = text.replace(".", "").replace(",", "").replace(":", "").replace(";", "").lower()
        text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        text = text.replace("'", "").replace('"', "").replace("`", "")
        text = text.replace("!", "").replace("?", "")
        text = text.replace("*", "").replace("+", "").replace("=", "")
        text = text.replace("{", "").replace("}", "")
        text = text.replace("|", "").replace("\\", "").replace("/", "").replace("~", "")
        text = text.replace("`", "").replace("´", "").replace("`", "").replace("‘", "")
        text = text.replace("“", "").replace("”", "").replace("'", "").replace("’", "")

        # Tokenize the text into words
        # words = text.split()
        tokenizer = MWETokenizer()

        keyword_processor = KeywordProcessor()

        keys = wv.index_to_key

        keyword_processor.add_keywords_from_list(keys)

        keywords_found = keyword_processor.extract_keywords(text, span_info=True)

        for a in keywords_found:
            tokenizer.add_mwe(text[a[1]: a[2]].split())

        tokens = tokenizer.tokenize(text.split())

        # Remove leading '<' and trailing '>' from each word (in case of iri)
        tokens = [word[1:-1] if word.startswith('<') and word.endswith('>') else word for word in tokens]
        # Remove leading "'<" and trailing ">'" from each word (in case of iri)
        tokens = [word[2:-2] if word.startswith("'<") and word.endswith(">'") else word for word in tokens]

        # Get the embeddings for each word
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
            # elif word2vec_model.wv.has_index_for(token):
            #     embeddings.append(word2vec_model.wv[token])
            else:
                # print(token.replace("_", ""))
                print(f"Token '{token}' not found in the model vocabulary.")
        # Return the mean of the embeddings
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(wv.vector_size)
        
    
    # Compute embeddings for each sentence in the dictionary
    sentence_embeddings = {key: get_owl2vec_embedding(sentence) for key, sentence in id_axiom_sentence_dict.items()}
    
    # Save embeddings to a file
    file_path = f"{ontology_json_file}_owl2vec_pretrained_embeddings.pkl"
    sentence_embeddings_converted = sentence_embeddings
    
    # Use TensorFlow's file I/O to save the numpy arrays
    with tf.io.gfile.GFile(file_path, "wb") as file:
        np.save(file, sentence_embeddings_converted)
    print(f"Embeddings saved to '{file_path}'.")
    
if __name__ == "__main__":
    main(sys.argv)