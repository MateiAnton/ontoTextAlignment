#!/usr/bin/env python3
"""
Example script for using OWL2Vec Star with BERT embeddings.
"""
import configparser
import os
import logging
import sys

from owl2vec_star.owl2vec_star import extract_owl2vec_model

# Set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 example_bert.py <ontology_file>")
        sys.exit(1)
    
    ontology_file = sys.argv[1]
    
    # Create a custom config file for BERT embeddings
    config = configparser.ConfigParser()
    
    # Basic section
    config['BASIC'] = {}
    config['BASIC']['ontology_file'] = ontology_file
    
    # Document section
    config['DOCUMENT'] = {}
    config['DOCUMENT']['cache_dir'] = './cache_bert'
    config['DOCUMENT']['ontology_projection'] = 'yes'
    config['DOCUMENT']['projection_only_taxonomy'] = 'no'
    config['DOCUMENT']['multiple_labels'] = 'yes'
    config['DOCUMENT']['avoid_owl_constructs'] = 'no'
    config['DOCUMENT']['save_document'] = 'yes'
    config['DOCUMENT']['axiom_reasoner'] = 'none'
    config['DOCUMENT']['walker'] = 'random'
    config['DOCUMENT']['walk_depth'] = '3'
    config['DOCUMENT']['URI_Doc'] = 'yes'
    config['DOCUMENT']['Lit_Doc'] = 'yes'
    config['DOCUMENT']['Mix_Doc'] = 'no'
    
    # Model section
    config['MODEL'] = {}
    config['MODEL']['embed_size'] = '768'  # Default size for BERT base
    config['MODEL']['embedding_model'] = 'bert'
    config['MODEL']['bert_model_name'] = 'bert-base-uncased'
    config['MODEL']['bert_pooling_strategy'] = 'mean'
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.makedirs(config['DOCUMENT']['cache_dir'])
    
    # Write config to file
    with open('bert_config.cfg', 'w') as configfile:
        config.write(configfile)
    
    logging.info(f"Using ontology file: {ontology_file}")
    logging.info("Extracting OWL2Vec* model with BERT embeddings...")
    
    # Extract the OWL2Vec* model with BERT embeddings
    try:
        model = extract_owl2vec_model(
            ontology_file=ontology_file,
            config_file='bert_config.cfg',
            uri_doc=True,
            lit_doc=True,
            mix_doc=False
        )
        
        if model:
            # Save the model
            output_path = os.path.join(config['DOCUMENT']['cache_dir'], 'owl2vec_bert_model')
            model.save(output_path)
            logging.info(f"Model saved to {output_path}")
            
            # Save in word2vec format for compatibility with other tools
            word2vec_path = os.path.join(config['DOCUMENT']['cache_dir'], 'owl2vec_bert_model.txt')
            model.save_word2vec_format(word2vec_path, binary=False)
            logging.info(f"Model saved in word2vec format to {word2vec_path}")
            
            # Example: print some entity vectors
            if len(model.index_to_key) > 0:
                entity = model.index_to_key[0]
                logging.info(f"Example entity: {entity}")
                logging.info(f"Vector dimension: {len(model[entity])}")
        else:
            logging.error("Failed to create the model.")
    except Exception as e:
        logging.error(f"Error during model extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
