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

model_7b = "meta-llama/Llama-2-7b-chat-hf"
# model_7b = "meta-llama/Llama-2-7b-chat-hf"
# id_to_axiom_sentence = read_dict_from_json('/data4T/jieying/data/ontologies/GeoFault/merge_geofault_xml_del_pt.owl_id_to_axiom_sentence.json')
# id_to_axiom_sentence = read_dict_from_json('/data4T/jieying/data/ontologies/foodon/foodon-merged.owl_id_to_axiom_sentence.json')


# file_path = 'GeoFaultBenchmark.pkl_finalAnalysedResult.pkl_llama.pkl_gpt.pkl'
# file_path = '/data4T/jieying/data/ontologies/foodon/benchmark/achive/benchmark_raw_grouped_manual_part_1.csv_withoutFAISS_finalAnalysedResult.pkl'

if len(sys.argv) < 1:
    raise ValueError("Insufficient arguments provided. Please provide the json_file_path, data file path, and GPU ID.")

# json_file_path = sys.argv[1]
# DIR = sys.argv[2]
file_path = sys.argv[1]
# axiom_index_file = sys.argv[4]
gpu_id = 0
# print(f'{DIR}AD/atom-axioms_v1.txt')


# Check if GPU is available and specify GPU ID
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



gc.collect()

# Read data
# id_to_axiom_sentence = read_dict_from_json(json_file_path)



tokenizer_7b = AutoTokenizer.from_pretrained(model_7b)

# Specify the device explicitly in the pipeline
pipeline_7b = transformers.pipeline(
    "text-generation",
    model=model_7b,
    torch_dtype=torch.float16,
    device=device,  # Use GPU if available, otherwise CPU
)

df = pd.read_pickle(file_path)
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
        sequences = pipeline_7b(
            llama_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer_7b.eos_token_id,
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

if 'onto enrich Answer_BERT_llama_7b' not in df.columns:
    df['onto enrich Answer_BERT_llama_7b'] = 'tba'
    df['onto enrich Answer_SBERT_llama_7b'] = 'tba'
    df['onto enrich Answer_SapBERT_llama_7b'] = 'tba'


# Save the DataFrame
final_output_file = file_path+"_onto_enriched_llama7b.pkl"

process_queries_in_chunks(df,'onto_enriched_BERT_prompt','onto enrich Answer_BERT_llama_7b',final_output_file)
process_queries_in_chunks(df,'onto_enriched_SBERT_prompt','onto enrich Answer_SBERT_llama_7b',final_output_file)
process_queries_in_chunks(df,'onto_enriched_SapBERT_prompt','onto enrich Answer_SapBERT_llama_7b',final_output_file)



df.to_pickle(final_output_file)

# Display the first few rows of the DataFrame to verify
df

