import sys
from owl2vec_star.owl2vec_star import extract_owl2vec_model

def main(argv):
    """
    This script processes a JSON file containing ontology axioms to calculate and save their embeddings.
    
    Args:
    argv (list): Command line arguments where argv[1] is the JSON file path.
    """
    if len(argv) != 2:
        print(f'Usage: {argv[0]} <path_to_ontology_file>')
        sys.exit(1)

    ontology_file = argv[1]

    # Embed ontology using owl2vec_star
    gensim_model = extract_owl2vec_model(ontology_file, "./default.cfg", True, True, True)
    output_folder="/var/scratch/man471/cache/output_pretrained"

    #Gensim format
    gensim_model.save(output_folder+"ontology.embeddings")
    #Txt format
    gensim_model.wv.save_word2vec_format(output_folder+"ontology.embeddings.txt", binary=False)


if __name__ == "__main__":
    main(sys.argv)

