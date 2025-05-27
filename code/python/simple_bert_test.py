#!/usr/bin/env python3
"""
Simple test for a single BERT variant in owl2vec_star with simplified approach.
Usage: python simple_bert_test.py [variant] [model_name]
Example: python simple_bert_test.py bert bert-base-uncased
"""

import os
import sys
import configparser
import logging
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Set default values
    variant = 'bert'
    model_name = 'bert-base-uncased'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        variant = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    
    logger.info(f"Testing {variant} variant with model: {model_name}")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists('./simple_test_cache'):
        os.makedirs('./simple_test_cache')
    
    # Initialize model based on variant
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = None
    tokenizer = None
    embed_func = None
    
    try:
        if variant == 'bert':
            logger.info(f"Initializing BERT model: {model_name}")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            
            def get_bert_embedding(text):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
                return embedding.cpu().numpy()[0]
                
            embed_func = get_bert_embedding
            embed_size = 768
            
        elif variant == 'bert-large':
            logger.info("Initializing BERT-Large model")
            bert_model_name = 'bert-large-uncased' if not model_name else model_name
            tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            model = BertModel.from_pretrained(bert_model_name).to(device)
            model.eval()
            
            def get_bert_large_embedding(text):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
                return embedding.cpu().numpy()[0]
                
            embed_func = get_bert_large_embedding
            embed_size = 1024
            
        elif variant == 'sbert':
            logger.info(f"Initializing SBERT model: {model_name}")
            model = SentenceTransformer(model_name).to(device)
            
            def get_sbert_embedding(text):
                embedding = model.encode(text, convert_to_tensor=True)
                return embedding.cpu().numpy()
                
            embed_func = get_sbert_embedding
            embed_size = model.get_sentence_embedding_dimension()
            
        elif variant == 'sapbert':
            logger.info("Initializing SapBERT model")
            if not model_name or model_name == 'default':
                model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            
            def get_sapbert_embedding(text):
                tokens = tokenizer.encode_plus(
                    text, 
                    add_special_tokens=True, 
                    max_length=512, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**tokens)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    
                return embedding.cpu().numpy()[0]
                
            embed_func = get_sapbert_embedding
            embed_size = 768
            
        else:
            logger.error(f"Unsupported variant: {variant}")
            return
            
        # Test with a few sample texts
        sample_texts = [
            "This is a test sentence",
            "Ontology alignment is the process of determining correspondences between concepts",
            "BERT models are transformer-based language models"
        ]
        
        embeddings = []
        for i, text in enumerate(sample_texts):
            logger.info(f"Generating embedding for sample text {i+1}")
            embedding = embed_func(text)
            embeddings.append(embedding)
            logger.info(f"Shape of embedding: {embedding.shape}")
            logger.info(f"First five dimensions: {embedding[:5]}")
            
        # Calculate similarity between embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        logger.info("Calculating cosine similarities between embeddings:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                logger.info(f"Similarity between text {i+1} and {j+1}: {sim:.4f}")
                
        logger.info(f"{variant.upper()} TEST: SUCCESS")
        
    except Exception as e:
        logger.error(f"Error testing {variant}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"{variant.upper()} TEST: FAILED")

if __name__ == "__main__":
    main()
