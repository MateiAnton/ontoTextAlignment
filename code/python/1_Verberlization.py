import sys
import json
from deeponto.onto import Ontology, OntologyVerbaliser

def create_mapping_axiom_sentence(onto, verbaliser):
    """Process ontology axioms to create mappings between axiom IDs, their string representations,
    and verbalized sentences.

    Args:
        onto (Ontology): The ontology object loaded from the file.
        verbaliser (OntologyVerbaliser): The verbaliser used to convert axioms to natural language sentences.

    Returns:
        tuple: Two dictionaries; the first maps IDs to axiom strings, and the second maps IDs to verbalized sentences.
    """
    id_to_axiom = {}
    id_to_axiom_sentence = {}
    i = 0

    # Process subsumption axioms
    for axiom in onto.get_subsumption_axioms():
        try:
            i += 1
            result = verbaliser.verbalise_class_subsumption_axiom(axiom)
            data = {
                'ID': i,
                'axiom': str(axiom),
                'Sentence': f"Every {result[0].verbal} is a {result[1].verbal}"
            }
            id_to_axiom[data['ID']] = data['axiom']
            id_to_axiom_sentence[data['ID']] = data['Sentence']
        except RuntimeError as e:
            print(f"Error processing subsumption axiom {axiom}: {e}")
            continue

    # Process equivalence axioms
    for axiom in onto.get_equivalence_axioms():
        try:
            i += 1
            result = verbaliser.verbalise_class_equivalence_axiom(axiom)
            data = {
                'ID': i,
                'axiom': str(axiom),
                'Sentence': f"The concept '{result[0].verbal}' is defined as {result[1].verbal}"
            }
            id_to_axiom[data['ID']] = data['axiom']
            id_to_axiom_sentence[data['ID']] = data['Sentence']
        except RuntimeError as e:
            print(f"Error processing equivalence axiom {axiom}: {e}")
            continue

    return id_to_axiom, id_to_axiom_sentence

def main(argv):
    """Main function to load an ontology, create mappings of axioms, and save them as JSON.

    Args:
        argv (list): Command line arguments where argv[0] is the path to the ontology file.

    Output:
        Saves two JSON files containing mappings from axiom IDs to their string representations and verbalized sentences.
    """
    ontology_file = argv[0]
    onto = Ontology(ontology_file)
    verbaliser = OntologyVerbaliser(onto)
    print("Finished loading the ontology and initializing the verbaliser.")

    id_to_axiom, id_to_axiom_sentence = create_mapping_axiom_sentence(onto, verbaliser)
    print(f"Processed {len(id_to_axiom)} axioms and generated sentences.")

    # Save the dictionaries as JSON files
    id_to_axiom_file_path = f"{ontology_file}_id_to_axiom.json"
    id_to_axiom_sentence_file_path = f"{ontology_file}_id_to_axiom_sentence.json"
    
    with open(id_to_axiom_file_path, 'w') as json_file:
        json.dump(id_to_axiom, json_file, indent=4)
    
    with open(id_to_axiom_sentence_file_path, 'w') as json_file2:
        json.dump(id_to_axiom_sentence, json_file2, indent=4)
    
    print("Dictionaries saved as JSON.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Syntax: {sys.argv[0]} <ontology_file>")
        sys.exit(0)
    main(sys.argv[1:])
