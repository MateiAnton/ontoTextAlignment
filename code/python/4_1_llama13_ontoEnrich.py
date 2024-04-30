import pandas as pd
from transformers import AutoTokenizer, pipeline
import transformers
import torch
from tqdm.auto import tqdm
import time
import os
import json
import sys
import gc
import math
import re

def read_dict_from_json(json_file_path):
    """
    Reads a dictionary from a JSON file.

    Parameters:
    json_file_path (str): Path to the JSON file.

    Returns:
    dict: Dictionary read from the JSON file, or None if an error occurred.
    """
    # Check if the file exists
    if not os.path.exists(json_file_path):
        print(f"Error: The file {json_file_path} does not exist.")
        return None

    try:
        # Open and read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError:
        print(f"Error: The file {json_file_path} does not contain valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

model_13b = "meta-llama/Llama-2-13b-chat-hf"
model_7b = "meta-llama/Llama-2-7b-chat-hf"
# id_to_axiom_sentence = read_dict_from_json('/data4T/jieying/data/ontologies/GeoFault/merge_geofault_xml_del_pt.owl_id_to_axiom_sentence.json')
# id_to_axiom_sentence = read_dict_from_json('/data4T/jieying/data/ontologies/foodon/foodon-merged.owl_id_to_axiom_sentence.json')


# file_path = 'GeoFaultBenchmark.pkl_finalAnalysedResult.pkl_llama.pkl_gpt.pkl'
# file_path = '/data4T/jieying/data/ontologies/foodon/benchmark/achive/benchmark_raw_grouped_manual_part_1.csv_withoutFAISS_finalAnalysedResult.pkl'

if len(sys.argv) < 4:
    raise ValueError("Insufficient arguments provided. Please provide the json_file_path, data file path, and GPU ID.")

json_file_path = sys.argv[1]
DIR = sys.argv[2]
file_path = sys.argv[3]
axiom_index_file = sys.argv[4]
gpu_id = 0
# print(f'{DIR}AD/atom-axioms_v1.txt')

# # Reading the text file
# with open(f'{DIR}AD/atom-axioms_v1.txt', 'r') as file:
#     text = file.read(file)

# Reading the JSON file
with open(axiom_index_file, 'r') as file:
    axiom_index = json.load(file)

# Reading the JSON file
with open(json_file_path, 'r') as file:
    axiom_sentence_index = json.load(file)


# Regex to extract atoms and axioms
atom_regex = r"(Atom \d+ has axioms:.+?)(?=\n\nAtom|\Z)"
axiom_regex = r": (.*?)(?=\n  - Axiom|\n\n|$)"

# Load the axiom ID index from JSON (Replace with actual file loading if needed)
# with open('axiom_index.json', 'r') as file:
#     axiom_index = json.load(file)

# Dictionary to hold atom IDs and their corresponding axiom IDs
atom_to_axiom_ids = {}
unfound_axioms = {}
new_axiom_id_counter = 1  # Start a counter for new axiom IDs


# Print the resulting dictionary
# print(atom_to_axiom_ids)

# Save the new unfound axiom index
if unfound_axioms:
    with open(os.path.join(DIR, 'unfound_axiom_ID.json'), 'w') as file:
        json.dump(unfound_axioms, file)

onto_graph_file = f'{DIR}AD/onto_graph.json'
with open(onto_graph_file, 'r') as file:
    onto_graph_edge = json.load(file)

# Check if GPU is available and specify GPU ID
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(file_path)
# Keeping only the first two rows
df = pd.read_pickle(file_path)

# Inverting the id_axiom to map labels to IDs
label_to_id = {label: id for id, label in axiom_index.items()}

# Create new edges_dict using IDs, check if both nodes exist
simple_onto_graph_edges_dict = {}
for start_label, end_labels in onto_graph_edge.items():
    if start_label in label_to_id:
        start_id = label_to_id[start_label]
        simple_onto_graph_edges_dict[start_id] = []
        for end_label in end_labels:
            if end_label in label_to_id:
                end_id = label_to_id[end_label]
                simple_onto_graph_edges_dict[start_id].append(end_id)

# Output the results
# print("Original edges_dict (by labels):", onto_graph_edge)
# print("New edges_dict (by IDs):", simple_onto_graph_edges_dict)

def dict_to_tuple_pairs(input_dict):
    """
    Converts a dictionary with iterable values into a tuple of key-value pairs,
    excluding pairs where the key or value starts with 'a'.

    Parameters:
    - input_dict (dict): The dictionary to convert, with iterables as values.

    Returns:
    - tuple: A tuple containing key-item pairs from the dictionary, excluding
             any key or value that starts with 'a'.
    """
    # Initialize an empty list to hold all valid key-value pairs
    all_pairs = []
    
    # Iterate through each key and iterable value in the dictionary
    for key, values in input_dict.items():
        # Skip if the key starts with 'a'
        if key.startswith('a'):
            continue

        # Check if value is iterable and not a string
        if hasattr(values, '__iter__') and not isinstance(values, str):
            # Extend the list with tuples pairing the key with each value, excluding values starting with 'a'
            all_pairs.extend((key, value) for value in values if not value.startswith('a'))
        else:
            # Handle the case where the value is not iterable or is a single string
            # Only add if the value does not start with 'a'
            if not values.startswith('a'):
                all_pairs.append((key, values))
    
    # Convert the list of pairs to a tuple
    tuple_pairs = tuple(all_pairs)
    
    return tuple_pairs

simple_onto_graph_edges_set = dict_to_tuple_pairs(simple_onto_graph_edges_dict)
# print(simple_onto_graph_edges_set)


def generate_onto_enriched_prompt_for_row(query, candidate_ids, id_to_sentence_map, bert_score_dict, onto_graph_edge, k):
    # Start the prompt
    prompt = "### Input:\n"
    prompt += "Could you please find the most relevant sentences from the following candidate sentences with respect to the reference sentence. Please provide the ranking in the format: 'The ranking of candidates is: [ranked list of IDs]. Please answer briefly using candidate IDs, separated by commas.'\n\n"
    prompt += f"Here is the reference sentence:\n\"{query}\"\n\n"
    prompt += "Candidates:\n\n"

    n = math.ceil(k / 2)
    # print(candidate_ids)
    # print(n)
    # print(candidate_ids[:n])
    # Get the initial set of IDs
    initial_set = {id if isinstance(id, str) else str(id) for id in candidate_ids[:n]}

    # print(initial_set)
    # Enrich the set by including related IDs from the graph set
    enriched_set = set(initial_set)
    for tuple_pair in onto_graph_edge:
        if tuple_pair[0] in initial_set or tuple_pair[1] in initial_set:
            enriched_set.update(tuple_pair)
    # print("-----",query)
    # for new_id in enriched_set-initial_set:
    #     if not new_id.lower().startswith("a"):
            # print (axiom_sentence_index[new_id])
    # Sort the enriched set
    sorted_enriched_set = sorted(enriched_set, key=lambda x: bert_score_dict.get(x, -float('inf')), reverse=True)
    # print(sorted_enriched_set)

    # Build the candidates list
    for id in sorted_enriched_set[:k]:
        sentence = id_to_sentence_map.get(id, f"Sentence not found for ID {id}")
        prompt += f"ID: {id} - \"{sentence}\"\n"

    prompt += "### Response:"
    return prompt

# Example usage in DataFrame
m=20
# df = df.head(2)
df['onto_enriched_BERT_prompt'] = df.apply(lambda row: generate_onto_enriched_prompt_for_row(
    row['Annotation Text'] if 'Annotation Text' in df.columns else row['Query'], 
    list(row['BERT_Ranking'].keys()),
    axiom_sentence_index,
    row['BERT_Ranking'],
    simple_onto_graph_edges_set,
    k=m), axis=1)
df['onto_enriched_SBERT_prompt'] = df.apply(lambda row: generate_onto_enriched_prompt_for_row(
    row['Annotation Text'] if 'Annotation Text' in df.columns else row['Query'], 
    list(row['SBERT_Ranking'].keys()),
    axiom_sentence_index,
    row['SBERT_Ranking'],
    simple_onto_graph_edges_set,
    k=m), axis=1)
df['onto_enriched_SapBERT_prompt'] = df.apply(lambda row: generate_onto_enriched_prompt_for_row(
    row['Annotation Text'] if 'Annotation Text' in df.columns else row['Query'], 
    list(row['SapBERT_Ranking'].keys()),
    axiom_sentence_index,
    row['SapBERT_Ranking'],
    simple_onto_graph_edges_set,
    k=m), axis=1)
# df['onto_enriched_BERT_prompt'][0]



# def generate_prompt_for_row(query, candidate_ids, id_to_sentence_map, k):
#     # Start the prompt with the query from the DataFrame
#     prompt = "### Input:"
#     prompt += "Could you please find the most relevant setences from the following candidate sentences with respect to the reference sentence. Please provide the ranking in the format: \"The ranking of candidates is: [ranked list of IDs].Please answer briefly using candidate IDs, separated by commas.\"\n\n"
#     prompt += f"Here is the reference sentence:\n\"{query}\"\n\n"
#     prompt += "Candidates:\n\n"

#     # Include up to 'k' candidate IDs in the prompt
#     for id in candidate_ids[:k]:  # Only iterate over the first 'k' IDs
#         sentence = id_to_sentence_map.get(id, "Sentence not found for ID {}".format(id))
#         prompt += f"ID: {id} - \"{sentence}\"\n"

#     prompt += "### Response:"
#     return prompt



gc.collect()

# Read data
id_to_axiom_sentence = read_dict_from_json(json_file_path)



tokenizer_13b = AutoTokenizer.from_pretrained(model_13b)

# Specify the device explicitly in the pipeline
pipeline_13b = transformers.pipeline(
    "text-generation",
    model=model_13b,
    torch_dtype=torch.float16,
    device=device,  # Use GPU if available, otherwise CPU
)

# tokenizer_7b = AutoTokenizer.from_pretrained(model_7b)

# # Specify the device explicitly in the pipeline
# pipeline_7b = transformers.pipeline(
#     "text-generation",
#     model=model_7b,
#     torch_dtype=torch.float16,
#     device=device,  # Use GPU if available, otherwise CPU
# )


# Function to generate response and measure computation time
def generate_response(query):
    start_time = time.time()
    llama_prompt = "[INST]" + query + "[/INST]"
    with torch.no_grad():
    # Place your inference code here
        sequences = pipeline_13b(
            llama_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer_13b.eos_token_id,
            max_length=9000,
        )
    end_time = time.time()
    computation_time = end_time - start_time
    # print(sequences[0]['generated_text'])
    return sequences[0]['generated_text'], computation_time


def process_queries_in_chunks(df, query_column, answer_column, final_output_file, chunk_size=5, batch_size=5, save_every=10):
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]

        for batch_start in range(0, len(chunk), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk))
            queries = chunk[query_column][batch_start:batch_end].tolist()

            batch_results = []
            for i, query in enumerate(queries):
                if chunk.iloc[i][answer_column] == 'tba':
                    response = generate_response(query)
                else:
                    response = chunk.iloc[i][answer_column]
                batch_results.append(response)

            for i, result in enumerate(batch_results):
                if(chunk_start + batch_start + 1<len(df)):
                    df[answer_column].iloc[chunk_start + batch_start + i] = result


            if (chunk_start + batch_start) % save_every == 0:
                df.to_pickle(final_output_file)
                del batch_results
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            gc.collect()
        gc.collect()
    
    # After processing a batch or at regular intervals
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df.to_pickle(final_output_file)

if 'onto enrich Answer_BERT_llama_13b' not in df.columns:
    df['onto enrich Answer_BERT_llama_13b'] = 'tba'
    df['onto enrich Answer_SBERT_llama_13b'] = 'tba'
    df['onto enrich Answer_SapBERT_llama_13b'] = 'tba'


# Save the DataFrame
final_output_file = file_path+"_onto_enriched_llama13b.pkl"

process_queries_in_chunks(df,'onto_enriched_BERT_prompt','onto enrich Answer_BERT_llama_13b',final_output_file)
process_queries_in_chunks(df,'onto_enriched_SBERT_prompt','onto enrich Answer_SBERT_llama_13b',final_output_file)
process_queries_in_chunks(df,'onto_enriched_SapBERT_prompt','onto enrich Answer_SapBERT_llama_13b',final_output_file)



df.to_pickle(final_output_file)

# Display the first few rows of the DataFrame to verify
df
