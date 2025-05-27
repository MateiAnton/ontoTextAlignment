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
            'ontology_file': '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault.ttl'
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
    
    # Verify the ontology file exists
    ontology_file = '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault.ttl'
    if not os.path.exists(ontology_file):
        logger.error(f"Ontology file not found: {ontology_file}")
        logger.info("Available ontology files:")
        for potential_file in [
            '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault.ttl',
            '/home/matei/projects/ontoTextAlignment/code/python/original_data/Merged_GeoFault.ttl',
            '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault_modified.ttl'
        ]:
            logger.info(f"  {potential_file}: {'EXISTS' if os.path.exists(potential_file) else 'NOT FOUND'}")
        return
        
    # Test each variant (limited parameters for quick testing)
    variants = [
        ('bert', 'bert-base-uncased'),
        ('bert-large', 'bert-large-uncased'),  # Explicitly set the model name
        ('sbert', 'paraphrase-MiniLM-L6-v2'),
        ('sapbert', 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext')  # Explicitly set the model name
    ]
    
    # Only test one model at a time to avoid CUDA memory issues
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in [v[0] for v in variants]:
        # Test only the specified variant
        variant_name = sys.argv[1]
        selected_variant = next((v for v in variants if v[0] == variant_name), None)
        if selected_variant:
            success = test_bert_variant(*selected_variant)
            logger.info(f"{selected_variant[0]}: {'SUCCESS' if success else 'FAILED'}")
        return
    
    # Test all variants sequentially
    results = {}
    for variant, model_name in variants:
        try:
            results[variant] = test_bert_variant(variant, model_name)
        except Exception as e:
            logger.error(f"Unexpected error testing {variant}: {str(e)}")
            results[variant] = False
        
    # Print summary
    logger.info("\n=== Test Results ===")
    for variant, success in results.items():
        logger.info(f"{variant}: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    main()
