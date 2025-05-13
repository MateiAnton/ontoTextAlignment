# script to remap ids in the benchmark file from the original to the new ids


import json
import pandas as pd


benchmark_file_path = "./modified_original_data/GeoFaultBenchmark.csv"
original_id_to_axiom_file_path = "./modified_original_data/merge_GeoFault.owl_id_to_polished_axiom.json"
new_id_to_axiom_file_path = "./generated_data/Merged_GeoFault_modified.ttl_id_to_axiom.json"


# load files
with open(original_id_to_axiom_file_path, 'r') as json_file:
    original_id_to_axiom = json.load(json_file)

with open(new_id_to_axiom_file_path, 'r') as json_file:
    new_id_to_axiom = json.load(json_file)

# load benchmark file into pandas datastore
benchmark_data = pd.read_csv(benchmark_file_path)

# one axiom list looks like : "['7', '48', '6']"
axiom_lists = benchmark_data['Relevant Axioms IDs'].str.replace("[", "").str.replace("]", "").str.split(", ")
# remove quotes
axiom_lists = [[axiom.replace("'", "") for axiom in axiom_list] for axiom_list in axiom_lists]
# reverse new_id_to_axiom
axiom_to_new_id = {v: k for k, v in new_id_to_axiom.items()}

new_axiom_lists = []
for axiom_list in axiom_lists:
    new_axiom_list = []
    for axiom_id in axiom_list:
        # if axiom_id not in ['112', '95']:
            axiom = original_id_to_axiom[axiom_id]
            new_id = axiom_to_new_id[axiom]
            new_axiom_list.append(new_id)
    new_axiom_lists.append(new_axiom_list)

benchmark_data['Relevant Axioms IDs'] = new_axiom_lists
benchmark_data.to_csv("./generated_data/new_GeoFaultBenchmark.csv", index=False)
# print("New axiom lists: ", new_axiom_lists)

print("Remapped benchmark file successfully.")

