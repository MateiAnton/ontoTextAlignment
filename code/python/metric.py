import pandas as pd

import re
import ast

def can_convert_to_dict(string):
    try:
        result = ast.literal_eval(string)
        return isinstance(result, dict)
    except (SyntaxError, ValueError):
        return False

def hit_at_k(predicted, correct_answers, k):
    """
    Calculate whether there is a hit at top k predictions.

    Parameters:
    predicted (list): The list of predicted items.
    correct_answers (list): The list of correct (relevant) items.
    k (int): The number of top items to consider for checking the hit.

    Returns:
    int: 1 if there's a hit in the top k predictions, 0 otherwise.
    """
    # Ensure k is not greater than the length of the predicted list
    k = min(k, len(predicted))

    # Check if any of the top k predicted items are in the correct answers
    for item in predicted[:k]:
        if item in correct_answers:
            return 1  # Hit found

    return 0  # No hit found

def mean_hit_at_k(predicted_list, correct_answers_list, k):
    """
    Calculate the mean hit at top k predictions across multiple prediction sets.

    Parameters:
    predicted_list (list of list): The list of lists where each inner list is a set of predicted items.
    correct_answers_list (list of list): The list of lists where each inner list is a set of correct (relevant) items.
    k (int): The number of top items to consider for checking the hit.

    Returns:
    float: The mean of the hit@k values across all provided sets.
    """
    hit = 0
    # Process each set of predictions and corresponding correct answers
    for predicted_item in  predicted_list[:k]:
        if predicted_item in correct_answers_list:
            hit = hit+1  # Hit found, break after first hit to avoid multiple counts
            break

     # We only consider the minimum of k or the number of correct answers to normalize hits correctly
    # max_hits_in_this_set = min(k, len(correct_answers_list))
    possible_hit = min(k, len(correct_answers_list))
        
    # Calculate the mean of hits
    mean_hits = hit / possible_hit if hit else 0
    return mean_hits


# def convert_string_to_number_list(input_data):
#     if isinstance(input_data, list):
#         # Assuming the list contains numbers or numeric strings
#         # print([int(num) for num in input_data])
#         return [int(num) for num in input_data]
#     elif isinstance(input_data, str):
#         # Splitting the string by commas and converting each element to an integer
#         return [int(num.strip()) for num in input_data.split(',') if num.strip().isdigit()]
#     else:
#         raise ValueError("Input data must be a string or a list")
    



# def convert_string_to_number_list(input_data):
#     if isinstance(input_data, list):
#         # Assuming the list contains numbers or numeric strings
#         # print([int(num) for num in input_data])
#         return [int(num) for num in input_data]
#     elif isinstance(input_data, str):
#         # Splitting the string by commas and converting each element to an integer
#         return [int(num.strip()) for num in input_data.split(',') if num.strip().isdigit()]
#     else:
#         raise ValueError("Input data must be a string or a list")
def convert_string_to_number_list(input_data):
    # print("Input received:", input_data)
    if isinstance(input_data, list):
        # print("it is a list")
        # Assuming the list contains numbers or numeric strings
        return [int(num) for num in input_data]
    elif isinstance(input_data, str):
        # print("it is a string")
        # Handling string formatted as a list or set
        if input_data.startswith('[') and input_data.endswith(']'):
            content = input_data[1:-1]
        elif input_data.startswith('{') and input_data.endswith('}'):
            content = input_data[1:-1]
        else:
            content = input_data
        
        # Process content, removing any potential quote characters
        processed_numbers = []
        for num in content.split(','):
            num_clean = num.strip().replace("'", "").replace("{", "").replace("}", "")
            if num_clean.isdigit():  # Ensure the string is numeric
                processed_numbers.append(int(num_clean))
        return processed_numbers
    elif isinstance(input_data, set):
        # print("it is a set")
        # Assuming the set contains numeric values directly
        try:
            return [int(num) for num in input_data]
        except ValueError:
            raise ValueError("All elements in the set must be integers or numeric strings.")
    else:
        print("Unsupported type received:", type(input_data))
        raise ValueError("Input data must be a string, a list, or a set")







# def gpt_convert_to_number_list(input_string):
#     # Find the start index of the first digit
#     # Check if the string contains any digits
#     if not any(c.isdigit() for c in input_string):
#         # Handle strings without digits (return an empty list or other appropriate value)
#         return []
    
#     start_index = next(i for i, c in enumerate(input_string) if c.isdigit())

#     # Find the end index of the last digit
#     end_index = next(i for i, c in reversed(list(enumerate(input_string))) if c.isdigit()) + 1

#     # Extract the substring that contains only the numbers
#     number_string = input_string[start_index:end_index]

#     # Split into individual number strings
#     number_strings = number_string.split(',')

#     # Convert each string to an integer
#     number_list = [int(num.strip()) for num in number_strings]

#     return number_list

def gpt_convert_to_number_list(input_string):
    # Split the input string by commas to separate potential number strings
    number_strings = input_string.split(',')

    # Initialize an empty list to store the integers
    number_list = []

    # Iterate over each part of the split string
    for num_str in number_strings:
        # Remove any leading/trailing whitespace
        num_str = num_str.strip()
        
        # Extract only the digits from each part
        digits = ''.join([char for char in num_str if char.isdigit()])
        
        # Check if the extracted string of digits is not empty
        if digits:
            # Convert the digits string to an integer
            number_list.append(int(digits))

    return number_list



def precision_at_k(predicted, reference, k):
    """
    Compute precision at k.

    Parameters:
    predicted (list): The list of predicted items.
    reference (list): The list of reference (relevant) items.
    k (int): The number of top items to consider for precision calculation.

    Returns:
    float: Precision at k.
    """
    if k == 0:
        return 0
    # Select the top-k items from the predicted list
    predicted_at_k = predicted[:k]
    # Count how many of the top-k items are in the reference list
    relevant_and_retrieved = sum(1 for item in predicted_at_k if item in reference)
    return relevant_and_retrieved / k



def extract_ranking_from_llama_text(text):
    """
    Extracts a list of numbers from the text following "### Response:[/INST]".
    Keeps the order of the numbers as they appear in the text.

    Args:
    text (str): The text from which to extract the numbers.

    Returns:
    list: A list of integers representing the extracted numbers.
    """
    # Find the substring following "### Response:[/INST]"

    if isinstance(text, str):
        new_text = text
    else:
        new_text = text[0]
    


    response_part = re.search(r'### Response:\[\/INST\](.*)', new_text, re.DOTALL)

    if response_part:
        response_text = response_part.group(1)

        # Find all numbers in the response text
        numbers = re.findall(r'\b\d+\b', response_text)

        # Convert found numbers to integers
        number_list = [int(num) for num in numbers]
        # print(number_list)
        return number_list
    else:
        return []
    

def extract_ranking_from_prompt(text):
    """
    Extracts a list of numbers from the text following "### Response:[/INST]".
    Keeps the order of the numbers as they appear in the text.

    Args:
    text (str): The text from which to extract the numbers.

    Returns:
    list: A list of integers representing the extracted numbers.
    """
   
    # Using regex to find all IDs
    numbers = re.findall(r'ID: (\d+)', text)

    # print(numbers)
    # Convert found numbers to integers
    # number_list = [int(numbers) for num in numbers]
    # print(number_list)
    return numbers
    


    


# def calculate_mean_hit_at_k(predicted_answers, correct_answers, k):
#     """
#     Calculate the mean hit at k for given lists of predicted and correct answers in a pandas DataFrame.

#     Parameters:
#     predicted_answers (list of list): List of lists containing predicted items.
#     correct_answers (list of list): List of lists containing correct items.
#     k (int): The number of top items to consider for checking the hit.

#     Returns:
#     float: The mean hit at k value across all rows of the DataFrame.
#     """

#     # Apply the hit_at_k function to each pair of predicted and correct answers
#     # hit_at_k_results = [hit_at_k(extract_ranking_from_llama_text(pred), convert_string_to_number_list(corr), k) for pred, corr in zip(predicted_answers, correct_answers)]
#     # Assuming the functions extract_ranking_from_llama_text and convert_string_to_number_list are defined elsewhere

#     # Initialize an empty list to store the results
#     hit_at_k_results = []

#     # Iterate over paired predicted and correct answers
#     for pred, corr in zip(predicted_answers, correct_answers):
#     # Process the predicted answer using extract_ranking_from_llama_text
#         processed_pred =[]
#         processed_corr =[]
        
#         if isinstance(pred,dict):
#             processed_pred = convert_string_to_number_list(list(pred.keys()))
#             # print(processed_pred)
#         else:
#             processed_pred = extract_ranking_from_llama_text(pred)
                

#         # Convert the correct answer string to a list of numbers
#         processed_corr = convert_string_to_number_list(corr)

        
#         # print(corr,processed_corr)
        
#         # Apply the hit_at_k function and append the result to hit_at_k_results
#         result = hit_at_k(processed_pred, processed_corr, k)
#         hit_at_k_results.append(result)

#     # Calculate and return the mean of the results
#     return sum(hit_at_k_results) / len(hit_at_k_results)
    
def calculate_mean_hit_at_k(predicted_answers, correct_answers, k, label):
    """
    Calculate the mean hit at k for given lists of predicted and correct answers in a pandas DataFrame.

    Parameters:
    predicted_answers (list of list/dict): List of lists or dicts containing predicted items.
    correct_answers (list of list): List of lists containing correct items.
    k (int): The number of top items to consider for checking the hit.
    label (str): Label indicating the type of processing to be applied on predicted answers. 
                 Options: 'embedding', 'gpt', 'llama'.

    Returns:
    float: The mean hit at k value across all rows of the DataFrame.
    """
    
    hit_at_k_results = []
    processed_pred =[]
    processed_corr =[]

    for pred, corr in zip(predicted_answers, correct_answers):
        # Check the type of predicted answer and process accordingly
        if isinstance(pred, dict):
            processed_pred = convert_string_to_number_list(list(pred.keys()))
        elif can_convert_to_dict(pred):
            processed_pred = convert_string_to_number_list(list(ast.literal_eval(pred).keys()))
            
        else:
            if label == 'embedding':
                processed_pred = convert_string_to_number_list(pred)
            elif label == 'gpt':
                processed_pred = gpt_convert_to_number_list(pred)
            elif label == 'llama':
                processed_pred = extract_ranking_from_llama_text(pred)
            else:
                raise ValueError("Invalid label. Choose from 'embedding', 'gpt', or 'llama'.")

        # Convert the correct answer string to a list of numbers
        processed_corr = convert_string_to_number_list(corr)
        # print(corr)
        # print(processed_corr)
        # print(processed_pred)
       
        # Apply the hit_at_k function and append the result
        result = mean_hit_at_k(processed_pred, processed_corr, k)
        hit_at_k_results.append(result)

    # Calculate and return the mean of the results
    return round(sum(hit_at_k_results) / len(hit_at_k_results),3)



def a_mrr(predicted, correct_answer):
    """
    Calculate the mean reciprocal rank.

    Parameters:
    predicted (list): The list of predicted items.
    correct_answers (list): The list of correct (relevant) items.

    Returns:
    float: The reciprocal rank of the first correct answer, 0 if none found.
    """
    for rank, item in enumerate(predicted, start=1):
        if item == correct_answer:
            return 1 / rank  # Reciprocal of the rank of the first correct answer

    return 0  # No correct answer found


def calculate_mean_mrr(predicted_answers, correct_answers, label):
    """
    Calculate the mean reciprocal rank (MRR) for given lists of predicted and correct answers in a pandas DataFrame.

    Parameters:
    predicted_answers (list of list/dict): List of lists or dicts containing predicted items.
    correct_answers (list of list): List of lists containing correct items.
    label (str): Label indicating the type of processing to be applied on predicted answers. 
                 Options: 'embedding', 'gpt', 'llama'.

    Returns:
    float: The mean reciprocal rank value across all rows of the DataFrame.
    """
    
    mrr_values = []

    for pred, corr in zip(predicted_answers, correct_answers):
        # Check the type of predicted answer and process accordingly
        processed_pred = []
        if isinstance(pred, dict):
            processed_pred = convert_string_to_number_list(list(pred.keys()))
        elif can_convert_to_dict(pred):
            processed_pred = convert_string_to_number_list(list(ast.literal_eval(pred).keys()))
        else:
            if label == 'embedding':
                processed_pred = convert_string_to_number_list(pred)
            elif label == 'gpt':
                processed_pred = gpt_convert_to_number_list(pred)
            elif label == 'llama':
                processed_pred = extract_ranking_from_llama_text(pred)
            else:
                raise ValueError("Invalid label. Choose from 'embedding', 'gpt', or 'llama'.")
            
        # Convert the correct answer string to a list of numbers
        processed_corr = convert_string_to_number_list(corr)

        # Calculate MRR for each correct answer in corr
        for correct_answer in processed_corr:
            mrr_score = a_mrr(processed_pred, correct_answer)
            # print(mrr_score,processed_pred,)
            mrr_values.append(mrr_score)

    # Calculate and return the mean of the MRR values
    return round(sum(mrr_values) / len(mrr_values), 3)


def eval_pro(df, flag):
    if 'ID of Relevant Axiom' in df.columns:
        column_name = 'ID of Relevant Axiom'
    elif 'Relevant Axioms IDs' in df.columns:
        column_name = 'Relevant Axioms IDs'
    else:
        column_name = 'union_IDs'

    ks = [1, 5, 10]  # Defined k values

    results = []
    for k in ks:
        model_name = 'Word2Vec_Ranking'
        if model_name in df.columns:
            hit_at_k_value = calculate_mean_hit_at_k(df[model_name], df[column_name], k, 'embedding')
            print(f'hit@{k} for word2vec ranking: {hit_at_k_value}')
            results.append(f'{hit_at_k_value}')
        else:
            print(f'Column {model_name} not found in DataFrame')
    if model_name in df.columns:
        mrr = calculate_mean_mrr(df[model_name], df[column_name], 'embedding')
        print(f'MRR for word2vec: {mrr}')
        results.append(f'{mrr}')
    print(' & '.join(results))
    print('**********')


    for bert in ['BERT', 'SBERT', 'SapBERT']:
        print('-------')
        results = []
        for k in ks:
            if f'{bert}_Ranking' in df.columns:
                hit_at_k_value = calculate_mean_hit_at_k(df[f'{bert}_Ranking'], df[column_name], k, 'embedding')
                print(f'hit@{k} for {bert} ranking: {hit_at_k_value}')
                results.append(f'{hit_at_k_value}')
            else:
                print(f'Column {bert}_Ranking not found in DataFrame')
        if f'{bert}_Ranking' in df.columns:
            mrr = calculate_mean_mrr(df[f'{bert}_Ranking'], df[column_name], 'llama')
            print(f'MRR for {bert}: {mrr}')
            results.append(f'{mrr}')
        print(' & '.join(results))
        print('-----')

    print('**********')

    results = []
    for k in ks:
        model_name = 'owl2vec_iri_Ranking'
        if model_name in df.columns:
            hit_at_k_value = calculate_mean_hit_at_k(df[model_name], df[column_name], k, 'embedding')
            print(f'hit@{k} for owl2vec with iri ranking: {hit_at_k_value}')
            results.append(f'{hit_at_k_value}')
        else:
            print(f'Column {model_name} not found in DataFrame')
    if model_name in df.columns:
        mrr = calculate_mean_mrr(df[model_name], df[column_name], 'embedding')
        print(f'MRR for owl2vec with iri: {mrr}')
        results.append(f'{mrr}')
    print(' & '.join(results))
    print('**********')

    results = []
    for k in ks:
        model_name = 'owl2vec_Ranking'
        if model_name in df.columns:
            hit_at_k_value = calculate_mean_hit_at_k(df[model_name], df[column_name], k, 'embedding')
            print(f'hit@{k} for owl2vec ranking: {hit_at_k_value}')
            results.append(f'{hit_at_k_value}')
        else:
            print(f'Column {model_name} not found in DataFrame')
    if model_name in df.columns:
        mrr = calculate_mean_mrr(df[model_name], df[column_name], 'embedding')
        print(f'MRR for owl2vec: {mrr}')
        results.append(f'{mrr}')
    print(' & '.join(results))
    print('**********')

    results = []
    for k in ks:
        model_name = 'owl2vec_pretrained_Ranking'
        if model_name in df.columns:
            hit_at_k_value = calculate_mean_hit_at_k(df[model_name], df[column_name], k, 'embedding')
            print(f'hit@{k} for owl2vec_pretrained ranking: {hit_at_k_value}')
            results.append(f'{hit_at_k_value}')
        else:
            print(f'Column {model_name} not found in DataFrame')
    if model_name in df.columns:
        mrr = calculate_mean_mrr(df[model_name], df[column_name], 'embedding')
        print(f'MRR for owl2vec_pretrained: {mrr}')
        results.append(f'{mrr}')
    print(' & '.join(results))
    print('**********')


    # for bert in ['BERT', 'SBERT', 'SapBERT']:
    #     for model in ['7b', '13b']:
    #         print('-------')
    #         results = []
    #         for k in ks:
    #             column_20 = f'Answer_{bert}_llama_{model}_20'
    #             column_10 = f'Answer_{bert}_llama_{model}_10'
    #             if column_20 in df.columns:
    #                 hit_at_k_value = calculate_mean_hit_at_k(df[column_20], df[column_name], k, 'llama')
    #                 print(f'hit@{k} for {bert} + {model}+ 20: {hit_at_k_value}')
    #                 results.append(f'{hit_at_k_value}')
    #             else:
    #                 print(f'Column {column_20} not found in DataFrame')
    #             if flag and k == 1 and column_10 in df.columns:
    #                 hit_at_k_value = calculate_mean_hit_at_k(df[column_10], df[column_name], 1, 'llama')
    #                 print(f'hit@{1} for {bert} + {model} + 10: {hit_at_k_value}')
    #                 results.append(f'{hit_at_k_value}')

    #         if column_20 in df.columns:
    #             mrr = calculate_mean_mrr(df[column_20], df[column_name], 'llama')
    #             print(f'MRR for {bert} + {model}+ 20: {mrr}')
    #             results.append(f'{mrr}')
    #         if flag and column_10 in df.columns:
    #             mrr = calculate_mean_mrr(df[column_10], df[column_name], 'llama')
    #             print(f'MRR for {bert} + {model} + 10: {mrr}')
    #             results.append(f'{mrr}')

    #         print(' & '.join(results))
    #         print('-----')
    #     print('**********')

    # print('**********')

    # for bert in ['BERT', 'SBERT', 'SapBERT']:
    #     for model in ['7b']:
    #         print('-------')
    #         results = []
    #         for k in ks:
    #             column_20 = f'Answer_{bert}_llama_{model}'
    #             if column_20 in df.columns:
    #                 hit_at_k_value = calculate_mean_hit_at_k(df[column_20], df[column_name], k, 'llama')
    #                 print(f'hit@{k} for {bert} + llama {model}: {hit_at_k_value}')
    #                 results.append(f'{hit_at_k_value}')
    #             else:
    #                 print(f'Column {column_20} not found in DataFrame')
                
    #         if  column_20 in df.columns:
    #             mrr = calculate_mean_mrr(df[column_20], df[column_name], 'llama')
    #             print(f'MRR for {bert} + llama {model}: {mrr}')
    #             results.append(f'{mrr}')

    #         print(' & '.join(results))
    #         print('-----')
    #     print('**********')

    #   # Enriched Columns
    # print("******Enriched*********")
    # enriched_keywords = ['onto enriched']  # Additional enriched categories
    # for keyword in enriched_keywords:
    #     columns_with_enrich = [col for col in df.columns if keyword in col.lower()]
    #     print(f'Columns related to {keyword}:', columns_with_enrich)
    #     if columns_with_enrich:
    #         for bert in ['BERT', 'SBERT', 'SapBERT']:
    #             for model in ['3.5', '4']:
    #                 print('-------')
    #                 results = []
    #                 for k in ks:
    #                     column_name_enriched = f'{keyword} Answer GPT {model} {bert}'
    #                     if column_name_enriched in df.columns:
    #                         hit_at_k_value = calculate_mean_hit_at_k(df[column_name_enriched], df[column_name], k, 'gpt')
    #                         print(f'hit@{k} for {bert} + enriched {model}: {hit_at_k_value}')
    #                         results.append(f'{hit_at_k_value}')
    #                     else:
    #                         print(f'Column {column_name_enriched} not found in DataFrame')
    #                 if results:
    #                     mrr = calculate_mean_mrr(df.get(column_name_enriched, pd.Series()), df[column_name], 'gpt')
    #                     print(f'MRR for {bert} + enriched {model}: {mrr}')
    #                     results.append(f'{mrr}')
    #                     print(' & '.join(results))
    #                 print('-----')
    #             print('**********')
    #     else:
    #         print('No enriched columns found for', keyword)
    
    # enriched_keywords = ['onto enriched']  # Additional enriched categories
    # for keyword in enriched_keywords:
    #     columns_with_enrich = [col for col in df.columns if keyword in col.lower()]
    #     print(f'Columns related to {keyword}:', columns_with_enrich)
    #     if columns_with_enrich:
    #         for bert in ['BERT', 'SBERT', 'SapBERT']:
    #             for model in ['7b', '13b']:
    #                 print('-------')
    #                 results = []
    #                 for k in ks:
    #                     column_20 = f'onto enrich Answer_{bert}_llama_{model}'
    #                     print('--------',column_20)
    #                     if column_20 in df.columns:
    #                         hit_at_k_value = calculate_mean_hit_at_k(df[column_20], df[column_name], k, 'llama')
    #                         print(f'hit@{k} for {bert} + llama + {model} + onto enrich: {hit_at_k_value}')
    #                         results.append(f'{hit_at_k_value}')
    #                     else:
    #                         print(f'Column {column_20} not found in DataFrame')

    #                 if column_20 in df.columns:
    #                     mrr = calculate_mean_mrr(df[column_20], df[column_name], 'llama')
    #                     print(f'MRR for {bert} + llama + {model} + onto enrich: {mrr}')
    #                     results.append(f'{mrr}')
    #                     print(' & '.join(results))
    #                 print('-----')
    #         print('**********')
    #     else:
    #         print('No enriched columns found for', keyword)

    # # GPT Section
    # print('****GPT******')
    # columns_with_gpt = [col for col in df.columns if 'gpt' in col.lower()]
    # if columns_with_gpt:
    #     for bert in ['BERT', 'SBERT', 'SapBERT']:
    #         for model in ['3.5', '4']:
    #             print('-------')
    #             results = []
    #             for k in ks:
    #                 column_name_gpt = f'Answer GPT {model} {bert}'
    #                 if column_name_gpt in df.columns:
    #                     hit_at_k_value = calculate_mean_hit_at_k(df[column_name_gpt], df[column_name], k, 'gpt')
    #                     print(f'hit@{k} for {bert} + GPT {model}: {hit_at_k_value}')
    #                     results.append(f'{hit_at_k_value}')
    #                 else:
    #                     print(f'Column {column_name_gpt} not found in DataFrame')
    #             if results:
    #                 mrr = calculate_mean_mrr(df.get(column_name_gpt, pd.Series()), df[column_name], 'gpt')
    #                 print(f'MRR for {bert} + GPT {model}: {mrr}')
    #                 results.append(f'{mrr}')
    #                 print(' & '.join(results))
    #             print('-----')
    #         print('**********')
    # else:
    #     print('No GPT-related columns found')


# Load the DataFrame
df = pd.read_csv('./generated_data/new_GeoFaultBenchmark_with_rankings.csv')

eval_pro(df, flag=False)
