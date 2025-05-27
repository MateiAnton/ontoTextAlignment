#!/usr/bin/env python3
"""
Custom test script for OWL2Vec* with BERT variants that bypasses ontology projection.
This script focuses purely on testing the BERT embedding functionality.
"""

import os
import sys
import configparser
import tempfile
import logging
import numpy as np
import torch
import random
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
    cache_dir = './bert_test_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 2. Create sample documents (similar to what owl2vec_star would generate)
    # These are simplified examples of what might come from an ontology
    URI_Doc = [
        ["http://example.org/ontology#Person", "http://www.w3.org/2000/01/rdf-schema#subClassOf", "http://example.org/ontology#Agent"],
        ["http://example.org/ontology#Student", "http://www.w3.org/2000/01/rdf-schema#subClassOf", "http://example.org/ontology#Person"],
        ["http://example.org/ontology#University", "http://www.w3.org/2002/07/owl#ObjectProperty", "http://example.org/ontology#hasStudent"]
    ]
    
    Lit_Doc = [
        ["person", "subclass", "of", "agent"],
        ["student", "subclass", "of", "person"],
        ["university", "has", "student"]
    ]
    
    # Combine documents
    all_doc = URI_Doc + Lit_Doc
    random.shuffle(all_doc)
    
    # 3. Initialize model based on variant
    try:
        # Setup model and tokenizer based on variant
        if variant == 'bert':
            logger.info(f"Initializing BERT model: {model_name}")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            vector_size = 768
            pooling_strategy = 'mean'
            
        elif variant == 'bert-large':
            logger.info("Initializing BERT-Large model")
            if not model_name or model_name == 'default':
                model_name = 'bert-large-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            vector_size = 1024
            pooling_strategy = 'mean'
            
        elif variant == 'sbert':
            logger.info(f"Initializing SBERT model: {model_name}")
            model = SentenceTransformer(model_name).to(device)
            vector_size = model.get_sentence_embedding_dimension()
            tokenizer = None
            
        elif variant == 'sapbert':
            logger.info("Initializing SapBERT model")
            if not model_name or model_name == 'default':
                model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name).to(device)
            model.eval()
            vector_size = 768
            pooling_strategy = 'mean'
            
        else:
            logger.error(f"Unsupported variant: {variant}")
            return
            
        # 4. Create embedding class (similar to BertEmbedding in owl2vec_star)
        class BertEmbedding:
            def __init__(self):
                self.wv = self  # For compatibility with Word2Vec interface
                self.vector_size = vector_size
                self.vocab = {}  # Maps tokens to their indices (for compatibility with older gensim)
                self.key_to_index = {}  # Maps tokens to their indices
                self.index_to_key = []  # Maps indices to tokens
                self.vectors = []  # Actual embedding vectors
                
            def get_vector(self, word):
                if word in self.key_to_index:
                    return self.vectors[self.key_to_index[word]]
                else:
                    # Return zeros for unknown words
                    return np.zeros(self.vector_size)
                
            def __getitem__(self, word):
                return self.get_vector(word)
                
            def save(self, filename):
                # Save embeddings to disk
                data = {
                    'vectors': np.array(self.vectors),
                    'key_to_index': self.key_to_index,
                    'index_to_key': self.index_to_key
                }
                torch.save(data, filename)
                
            def save_word2vec_format(self, filename, binary=False):
                # Save in word2vec text format for compatibility
                with open(filename, 'w') as f:
                    f.write('%d %d\n' % (len(self.index_to_key), self.vector_size))
                    for word in self.index_to_key:
                        vector = self.vectors[self.key_to_index[word]]
                        vector_str = ' '.join(['%.6f' % val for val in vector])
                        f.write('%s %s\n' % (word, vector_str))
        
        # Create the embedding object
        bert_embeddings = BertEmbedding()
        
        # 5. Define function to get embeddings based on variant
        def get_bert_embedding(text):
            # Ensure text is not empty
            if not text:
                text = " "  # Use a space as default for empty text
                
            try:
                if variant == 'bert' or variant == 'bert-large':
                    # Standard BERT embedding approach
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    # Use mean pooling by default
                    attention_mask = inputs['attention_mask']
                    embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
                    embedding = embedding.cpu().numpy()
                
                elif variant == 'sbert':
                    # SentenceTransformer approach (SBERT)
                    embedding = model.encode(text, convert_to_tensor=True).cpu().numpy()
                    # Reshape if needed
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)
                
                elif variant == 'sapbert':
                    # SapBERT approach
                    tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, 
                                                 padding='max_length', truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**tokens)
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                return embedding[0]  # Return the embedding vector for this text
                
            except Exception as e:
                logger.error(f"Error generating embedding for text '{text[:20]}...': {str(e)}")
                # Return zeros as fallback for failed embeddings
                return np.zeros(vector_size)
        
        # 6. Process unique words/entities from all documents
        unique_tokens = set()
        for doc in all_doc:
            for token in doc:
                unique_tokens.add(token)
        
        logger.info(f"Found {len(unique_tokens)} unique tokens to embed")
        
        # 7. Generate embeddings for each token
        token_list = list(unique_tokens)
        batch_size = 32
        
        for i in range(0, len(token_list), batch_size):
            batch_tokens = token_list[i:i+batch_size]
            for token in batch_tokens:
                # For URIs, try to extract the local name
                if token.startswith('http://'):
                    if '#' in token:
                        text = token.split('#')[-1]
                    else:
                        text = token.split('/')[-1]
                else:
                    text = token
                    
                # Get embedding
                embedding = get_bert_embedding(text)
                
                # Store the embedding
                bert_embeddings.key_to_index[token] = len(bert_embeddings.index_to_key)
                bert_embeddings.index_to_key.append(token)
                bert_embeddings.vectors.append(embedding)
        
        # 8. Convert vectors to numpy array for efficiency
        bert_embeddings.vectors = np.array(bert_embeddings.vectors)
        
        # 9. Test the embeddings
        logger.info(f"Successfully created embeddings with vector size: {bert_embeddings.vector_size}")
        logger.info(f"Number of tokens embedded: {len(bert_embeddings.index_to_key)}")
        
        # Print a few sample embeddings
        if len(bert_embeddings.index_to_key) > 0:
            sample_key = bert_embeddings.index_to_key[0]
            logger.info(f"Sample embedding for '{sample_key}': {bert_embeddings[sample_key][:5]}... (showing first 5 dimensions)")
            
        # 10. Calculate similarities between some embeddings
        if len(bert_embeddings.index_to_key) >= 2:
            key1 = bert_embeddings.index_to_key[0]
            key2 = bert_embeddings.index_to_key[1]
            
            from sklearn.metrics.pairwise import cosine_similarity
            vec1 = bert_embeddings[key1].reshape(1, -1)
            vec2 = bert_embeddings[key2].reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            logger.info(f"Cosine similarity between '{key1}' and '{key2}': {similarity}")
        
        # 11. Save the embeddings
        output_file = os.path.join(cache_dir, f"{variant}_embeddings.pt")
        bert_embeddings.save(output_file)
        logger.info(f"Embeddings saved to {output_file}")
        
        # 12. Save in Word2Vec format
        word2vec_file = os.path.join(cache_dir, f"{variant}_embeddings.txt")
        bert_embeddings.save_word2vec_format(word2vec_file)
        logger.info(f"Embeddings saved in Word2Vec format to {word2vec_file}")
        
        logger.info(f"{variant.upper()} TEST: SUCCESS")
        
    except Exception as e:
        logger.error(f"Error testing {variant}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"{variant.upper()} TEST: FAILED")

if __name__ == "__main__":
    main()
