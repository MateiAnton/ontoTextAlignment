import argparse
import json
import re
import os

def generate_semantic_atom_graphs(atom_to_axiom_ids):
    atom_graphs = {}
    for axioms in atom_to_axiom_ids.values():
        for axiom_from in axioms:
            for axiom_to in axioms:
                if axiom_from != axiom_to:
                    edge = (axiom_from, axiom_to)
                    atom_graphs[edge] = atom_graphs.get(edge, 0) + 1
    return set(atom_graphs.keys())

def generate_simple_ontology_syntax_edges_dict(axiom_index, onto_graph_edge):
    label_to_id = {label: id for id, label in axiom_index.items()}
    simple_onto_graph_edges_dict = {}
    for start_label, end_labels in onto_graph_edge.items():
        if start_label in label_to_id:
            start_id = label_to_id[start_label]
            simple_onto_graph_edges_dict[start_id] = [
                label_to_id[end_label] for end_label in end_labels if end_label in label_to_id
            ]
    return simple_onto_graph_edges_dict

def generate_atom_dependency_edges(simple_onto_graph_edges_dict, atom_to_axiom_ids):
    generated_edges_dict = {}
    for dependent_atom, dependent_axioms in atom_to_axiom_ids.items():
        for dep_axiom in dependent_axioms:
            target_axioms = simple_onto_graph_edges_dict.get(dep_axiom, [])
            for target_axiom in target_axioms:
                for depended_atom, depended_axioms in atom_to_axiom_ids.items():
                    if target_axiom in depended_axioms:
                        edge_key = (dependent_atom, depended_atom)
                        generated_edges_dict[edge_key] = generated_edges_dict.get(edge_key, 0) + 1
    return set(generated_edges_dict.keys())

def main(DIR, axiom_file, anatomy_file):
    atom_regex = r"(Atom \d+ has axioms:.+?)(?=\n\nAtom|\Z)"
    axiom_regex = r": (.*?)(?=\n  - Axiom|\n\n|$)"

    # Read atom-axiom text from the file
    with open(f'{DIR}/atom-axioms_v1.txt', 'r') as file:
        text = file.read()

    # Load the axiom ID index from JSON
    with open(axiom_file, 'r') as file:
        axiom_index = json.load(file)

    # Generate atom_to_axiom_ids
    atom_to_axiom_ids = {}
    for atom_block in re.finditer(atom_regex, text, re.DOTALL):
        atom_id = re.search(r"Atom (\d+)", atom_block.group()).group(1)
        axioms = re.findall(axiom_regex, atom_block.group(), re.MULTILINE)
        axiom_ids = set()
        for axiom in axioms:
            axiom = axiom.strip().replace(") )", "))")
            for key, value in axiom_index.items():
                if axiom == value:
                    axiom_ids.add(key)
        atom_to_axiom_ids[atom_id] = axiom_ids

    print(f"Generated atom to axiom IDs: {atom_to_axiom_ids}")

    # Continue with other processing
    with open(f'{DIR}/onto_graph.json', 'r') as file:
        onto_graph_edge = json.load(file)

    atom_graphs = generate_semantic_atom_graphs(atom_to_axiom_ids)
    onto_syntax_edges_dict = generate_simple_ontology_syntax_edges_dict(axiom_index, onto_graph_edge)
    atom_dependency_edges = generate_atom_dependency_edges(onto_syntax_edges_dict, atom_to_axiom_ids)

    print(atom_graphs)
    print(atom_dependency_edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ontology files to compute semantic graphs.")
    parser.add_argument("DIR", type=str, help="Directory containing input files")
    parser.add_argument("axiom_file", type=str, help="Path to the axiom index file")
    parser.add_argument("anatomy_file", type=str, help="Path to the anatomy axiom IDs file")
    args = parser.parse_args()

    main(args.DIR, args.axiom_file, args.anatomy_file)
