#!/usr/bin/env python3
"""
Simple test for a single BERT variant in owl2vec_star.
Usage: python test_single_bert_variant.py [variant] [model_name]
Example: python test_single_bert_variant.py bert bert-base-uncased
"""

import os
import sys
import configparser
import tempfile
import logging
from owl2vec_star.owl2vec_star import extract_owl2vec_model

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
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    
    # Use OWL format instead of TTL for better compatibility
    ontology_file = '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault.owl'
    if not os.path.exists(ontology_file):
        logger.error(f"Error: Ontology file not found at {ontology_file}")
        # Try to find alternative ontology files
        for potential_path in [
            '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault_modified.owl',
            '/home/matei/projects/ontoTextAlignment/code/python/original_data/Merged_GeoFault.owl',
            '/home/matei/projects/ontoTextAlignment/code/python/modified_original_data/Merged_GeoFault.ttl'
        ]:
            if os.path.exists(potential_path):
                ontology_file = potential_path
                logger.info(f"Found alternative ontology file: {ontology_file}")
                break
        else:
            logger.error("No valid ontology file found. Exiting.")
            return
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as tmp:
        config = configparser.ConfigParser()
        
        # Basic settings
        config['BASIC'] = {
            'ontology_file': ontology_file
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
            'bert_model_name': model_name,
            'bert_pooling_strategy': 'mean',
            'embed_size': '768',  # For BERT models
            'window': '5',
            'min_count': '1',
            'negative': '25',
            'seed': '42',
            'iteration': '5'  # Small number for testing
        }
        
        config.write(tmp)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Starting extraction with {variant} model...")
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
            
        logger.info(f"{variant} TEST: SUCCESS")
        
    except Exception as e:
        logger.error(f"Error testing {variant}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"{variant} TEST: FAILED")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    main()
