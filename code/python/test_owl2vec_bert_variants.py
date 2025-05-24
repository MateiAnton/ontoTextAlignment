#!/usr/bin/env python3
"""
Test script to validate the enhanced owl2vec_star library with new BERT variants.
This script tests each BERT variant implementation (BERT, BERT-Large, SBERT, SapBERT).
"""

import os
import configparser
import tempfile
import logging
from owl2vec_star.owl2vec_star import extract_owl2vec_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bert_variant(variant, model_name=None):
    """Test a specific BERT variant."""
    logger.info(f"Testing {variant} variant with model: {model_name}")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as tmp:
        config = configparser.ConfigParser()
        
        # Basic settings
        config['BASIC'] = {
            'ontology_file': '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault_modified.ttl'
        }
        
        # Document settings
        config['DOCUMENT'] = {
            'cache_dir': './cache',
            'URI_Doc': 'yes',
            'Lit_Doc': 'yes', 
            'Mix_Doc': 'no',
            'walker': 'random',
            'walk_depth': '2',
            'ontology_projection': 'yes',
            'projection_only_taxonomy': 'no',
            'multiple_labels': 'yes',
            'save_document': 'no'
        }
        
        # Model settings
        config['MODEL'] = {
            'embedding_model': variant,
            'embed_size': '100' if variant == 'word2vec' else '768',
            'window': '5',
            'min_count': '1',
            'negative': '25',
            'seed': '42',
            'iteration': '5'  # Small number for testing
        }
        
        # Add specific BERT model name if provided
        if model_name:
            config['MODEL']['bert_model_name'] = model_name
            
        config.write(tmp)
        tmp_path = tmp.name
    
    try:
        # Extract the model
        model = extract_owl2vec_model(
            ontology_file=config['BASIC']['ontology_file'],
            config_file=tmp_path,
            uri_doc=True, 
            lit_doc=True,
            mix_doc=False
        )
        
        # Test if the model was created successfully
        logger.info(f"Model created successfully with vector size: {model.vector_size}")
        logger.info(f"Number of tokens embedded: {len(model.index_to_key)}")
        
        # Print a few sample embeddings
        if len(model.index_to_key) > 0:
            sample_key = model.index_to_key[0]
            logger.info(f"Sample embedding for '{sample_key}': {model[sample_key][:5]}... (showing first 5 dimensions)")
            
        return True
    except Exception as e:
        logger.error(f"Error testing {variant}: {str(e)}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def main():
    """Test all BERT variants."""
    
    # Create cache dir if it doesn't exist
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
        
    # Test each variant (limited parameters for quick testing)
    variants = [
        ('bert', 'bert-base-uncased'),
        ('bert-large', None),  # Uses default bert-large-uncased
        ('sbert', 'paraphrase-MiniLM-L6-v2'),
        ('sapbert', None)  # Uses default SapBERT model
    ]
    
    results = {}
    for variant, model_name in variants:
        results[variant] = test_bert_variant(variant, model_name)
        
    # Print summary
    logger.info("\n=== Test Results ===")
    for variant, success in results.items():
        logger.info(f"{variant}: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    main()
