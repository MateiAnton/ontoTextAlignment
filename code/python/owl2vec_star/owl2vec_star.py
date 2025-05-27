"""Main module."""
import configparser
import multiprocessing
import os
import random
import sys
import time
import click
import logging
import numpy as np

import gensim
from owl2vec_star.lib.RDF2Vec_Embed import get_rdf2vec_walks
from owl2vec_star.lib.Label import pre_process_words, URI_parse
from owl2vec_star.lib.Onto_Projection import Reasoner, OntologyProjection

import nltk
nltk.download('punkt')    # Import transformers for BERT variants
try:
    import torch
    from transformers import BertModel, BertTokenizer
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("Transformers or torch library not found. BERT embeddings will not be available.")


'''
Main method to be called from libraries
'''
def extract_owl2vec_model(ontology_file, config_file, uri_doc, lit_doc, mix_doc):
    config = configparser.ConfigParser()
    config.read(click.format_filename(config_file))

    if ontology_file:
        config['BASIC']['ontology_file'] = click.format_filename(ontology_file)

    # Read embedding_model from config
    embedding_model = config.get('MODEL', 'embedding_model', fallback='word2vec')

    if uri_doc:
        config['DOCUMENT']['URI_Doc'] = 'yes'
    if lit_doc:
        config['DOCUMENT']['Lit_Doc'] = 'yes'
    if mix_doc:
        config['DOCUMENT']['Mix_Doc'] = 'yes'
    if 'cache_dir' not in config['DOCUMENT']:
        config['DOCUMENT']['cache_dir'] = './cache'

    if not os.path.exists(config['DOCUMENT']['cache_dir']):
        os.mkdir(config['DOCUMENT']['cache_dir'])    
    if embedding_model == 'word2vec':
        model_ = __perform_ontology_embedding(config)
    elif embedding_model in ['bert', 'bert-large', 'sbert', 'sapbert']:
        if not BERT_AVAILABLE:
            raise ImportError("BERT embeddings require torch and transformers libraries. Please install them with: pip install torch transformers sentence-transformers")
        model_ = __perform_bert_ontology_embedding(config)
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
        

    return model_



'''
Embedding of a single input ontology
'''
def __perform_ontology_embedding(config):

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    start_time = time.time()
    
    if ('ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes') or \
        'pre_entity_file' not in config['DOCUMENT'] or 'pre_axiom_file' not in config['DOCUMENT'] or \
        'pre_annotation_file' not in config['DOCUMENT']:
        logging.info('Access the ontology ...')

        tax_only = (config['DOCUMENT']['projection_only_taxonomy'] == "yes")
        
        projection = OntologyProjection(config['BASIC']['ontology_file'], reasoner=Reasoner.STRUCTURAL,
                                        only_taxonomy=tax_only,
                                        bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                        additional_preferred_labels_annotations=set(),
                                        additional_synonyms_annotations=set(),
                                        memory_reasoner='13351')
    else:
        projection = None

    # Ontology projection
    if 'ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes':
        logging.info('Calculate the ontology projection ...')
        projection.extractProjection()
        onto_projection_file = os.path.join(config['DOCUMENT']['cache_dir'], 'projection.ttl')
        projection.saveProjectionGraph(onto_projection_file)
        ontology_file = onto_projection_file
    else:
        ontology_file = config['BASIC']['ontology_file']

    # Extract and save seed entities (classes and individuals)
    # Or read entities specified by the user
    if 'pre_entity_file' in config['DOCUMENT']:
        entities = [line.strip() for line in open(config['DOCUMENT']['pre_entity_file']).readlines()]
    else:
        logging.info('Extract classes and individuals ...')
        projection.extractEntityURIs()
        classes = projection.getClassURIs()
        individuals = projection.getIndividualURIs()
        entities = classes.union(individuals)
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'entities.txt'), 'w') as f:
            for e in entities:
                f.write('%s\n' % e)

    # Extract axioms in Manchester Syntax if it is not pre_axiom_file is not set
    if 'pre_axiom_file' not in config['DOCUMENT']:
        logging.info('Extract axioms ...')
        projection.createManchesterSyntaxAxioms()
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt'), 'w') as f:
            for ax in projection.axioms_manchester:
                f.write('%s\n' % ax)

    # If pre_annotation_file is set, directly read annotations
    # else, read annotations including rdfs:label and other literals from the ontology
    #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
    #                        2) None label annotations as sentences of the literal document
    uri_label, uri_to_labels, annotations = dict(), dict(), list()

    if 'pre_annotation_file' in config['DOCUMENT']:
        with open(config['DOCUMENT']['pre_annotation_file']) as f:
            for line in f.readlines():
                tmp = line.strip().split()
                if tmp[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                    uri_label[tmp[0]] = pre_process_words(tmp[2:])
                else:
                    annotations.append([tmp[0]] + tmp[2:])

    else:
        logging.info('Extract annotations ...')
        projection.indexAnnotations()
        for e in entities:
            if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                label = list(projection.entityToPreferredLabels[e])[0]
                #Keeps only one
                uri_label[e] = pre_process_words(words=label.split())



                ##Populates dictionary with all labels per entity
                for label in projection.getPreferredLabelsForEntity(e):
                    #print("Preferred: " + label)
                    if e not in uri_to_labels:
                        uri_to_labels[e]=set()
                    #We add a list of words in the set
                    #print(pre_process_words(words=label.split()))
                    #print(uri_to_labels[e])
                    uri_to_labels[e].add(tuple(pre_process_words(words=label.split())))
		
                if e in projection.entityToSynonyms and len(projection.entityToSynonyms[e]) > 0:
                    for label in projection.getSynonymLabelsForEntity(e):
                        #print("Syn: " + label)
                        if e not in uri_to_labels:
                            uri_to_labels[e]=set()
                        #We add a list of words in the set
                        uri_to_labels[e].add(tuple(pre_process_words(words=label.split())))
                    
        for e in entities:
            if e in projection.entityToAllLexicalLabels:
                for v in projection.entityToAllLexicalLabels[e]:
                    if (v is not None) and \
                        (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[e])):
                        annotation = [e] + v.split()
                        annotations.append(annotation)

        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'annotations.txt'), 'w') as f:
            for e in projection.entityToPreferredLabels:
                for v in projection.entityToPreferredLabels[e]:
                    f.write('%s preferred_label %s\n' % (e, v))
            for a in annotations:
                f.write('%s\n' % ' '.join(a))

    # read URI document
    # two parts: walks, axioms (if the axiom file exists)
    walk_sentences, axiom_sentences, URI_Doc = list(), list(), list()
    if 'URI_Doc' in config['DOCUMENT'] and config['DOCUMENT']['URI_Doc'] == 'yes':
        logging.info('Generate URI document ...')
        walks_ = get_rdf2vec_walks(onto_file=ontology_file, walker_type=config['DOCUMENT']['walker'],
                                   walk_depth=int(config['DOCUMENT']['walk_depth']), classes=entities)
        logging.info('Extracted %d walks for %d seed entities' % (len(walks_), len(entities)))
        walk_sentences += [list(map(str, x)) for x in walks_]

        axiom_file = os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt')
        if os.path.exists(axiom_file):
            for line in open(axiom_file).readlines():
                axiom_sentence = [item for item in line.strip().split()]
                axiom_sentences.append(axiom_sentence)
        logging.info('Extracted %d axiom sentences' % len(axiom_sentences))
        URI_Doc = walk_sentences + axiom_sentences

    # Some entities have English labels
    # Keep the name of built-in properties (those starting with http://www.w3.org)
    # Some entities have no labels, then use the words in their URI name
    def label_item(item):
        if item in uri_label:
            return uri_label[item]
        elif item.startswith('http://www.w3.org'):
            return [item.split('#')[1].lower()]
        elif item.startswith('http://'):
            return URI_parse(uri=item)
        else:
            return [item.lower()]



    #New algorithm for multiple labels
    def getExtendedSentences(sentence, syn_dict, max_labels=5):
        sentences = list()
        tmp_sentences = list()

        for i, entity in enumerate(sentence):
            #print(i)
            if entity in syn_dict:
                for j, l in enumerate(syn_dict[entity]):
                    if j > max_labels:
                        break
                    # Initialization
                    if (i == 0):
                        sentences = sentences + [l]  # "l" already as a list
                    else:
                        for s in sentences:
                            s = s + l  # already as a list of words
                            tmp_sentences = tmp_sentences + [s]
                            # print(s)
            else:
                #For cases not in dictionary like OWL constructs
                for s in sentences:
                    s = s + tuple(label_item(entity))  # already as a list of words
                    tmp_sentences = tmp_sentences + [s]
                    # print(s)

            if (i > 0):
                sentences.clear()
                sentences = [s for s in tmp_sentences]
                tmp_sentences.clear()


        return sentences
    #End algorithm multiple labels


    # read literal document
    # two parts: literals in the annotations (subject's label + literal words)
    #            replacing walk/axiom sentences by words in their labels
    Lit_Doc = list()
    if 'Lit_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Lit_Doc'] == 'yes':
        logging.info('Generate literal document ...')
        for annotation in annotations:
            processed_words = pre_process_words(annotation[1:])
            if len(processed_words) > 0:
                Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
        logging.info('Extracted %d annotation sentences' % len(Lit_Doc))


        #Only applied to the walks?
        if 'multiple_labels' in config['DOCUMENT'] and config['DOCUMENT']['multiple_labels'] == "yes":
            for sentence in walk_sentences:
                for lit_sentence in getExtendedSentences(sentence, uri_to_labels, 5):
                    Lit_Doc.append(lit_sentence)


        else: #Single label
            for sentence in walk_sentences:
                lit_sentence = list()
                for item in sentence:
                    lit_sentence += label_item(item=item)
                Lit_Doc.append(lit_sentence)




        for sentence in axiom_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

    # read mixture document
    # for each axiom/walk sentence
    #   -    all): for each entity, keep its entity URI, replace the others by label words
    #   - random): randomly select one entity, keep its entity URI, replace the others by label words
    Mix_Doc = list()
    if 'Mix_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Mix_Doc'] == 'yes':
        logging.info('Generate mixture document ...')
        for sentence in walk_sentences + axiom_sentences:
            if config['DOCUMENT']['Mix_Type'] == 'all':
                for index in range(len(sentence)):
                    mix_sentence = list()
                    for i, item in enumerate(sentence):
                        mix_sentence += [item] if i == index else label_item(item=item)
                    Mix_Doc.append(mix_sentence)
            elif config['DOCUMENT']['Mix_Type'] == 'random':
                random_index = random.randint(0, len(sentence) - 1)
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    mix_sentence += [item] if i == random_index else label_item(item=item)
                Mix_Doc.append(mix_sentence)

    logging.info('URI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
    all_doc = URI_Doc + Lit_Doc + Mix_Doc

    logging.info('Time for document construction: %s seconds' % (time.time() - start_time))
    random.shuffle(all_doc)


    #Save all_doc (optional): default: no
    if config['DOCUMENT']['save_document'] == 'yes':
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'document_sentences.txt'), 'w') as f:
            for sentence in all_doc:
                for w in sentence:
                    f.write('%s ' % w)
                f.write('\n')
            f.close()


    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
        logging.info('Train the language model ...')
        model_ = gensim.models.Word2Vec(all_doc, vector_size=int(config['MODEL']['embed_size']),
                                        window=int(config['MODEL']['window']),
                                        workers=multiprocessing.cpu_count(),
                                        sg=1, epochs=int(config['MODEL']['iteration']),
                                        negative=int(config['MODEL']['negative']),
                                        min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
    else:
        logging.info('Fine-tune the pre-trained language model ...')
        model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
        if len(all_doc) > 0:
            model_.min_count = int(config['MODEL']['min_count'])
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))
	
    logging.info('Time for learning the language model: %s seconds' % (time.time() - start_time))

    return model_
    
    
    
    
'''
Joint embeddings with multiple input ontologies
'''
def __perform_joint_ontology_embedding(config):
    
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    start_time = time.time()

    walk_sentences, axiom_sentences = list(), list()
    uri_label, annotations = dict(), list()
    for file_name in os.listdir(config['BASIC']['ontology_dir']):
        if not file_name.endswith('.owl'):
            continue
        ONTO_FILE = os.path.join(config['BASIC']['ontology_dir'], file_name)
        logging.info('\nProcessing %s' % file_name)
        projection = OntologyProjection(ONTO_FILE, reasoner=Reasoner.STRUCTURAL, only_taxonomy=False,
                                        bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                        additional_preferred_labels_annotations=set(),
                                        additional_synonyms_annotations=set(), memory_reasoner='13351')

        # Extract and save seed entities (classes and individuals)
        logging.info('... Extract entities (classes and individuals) ...')
        projection.extractEntityURIs()
        classes = projection.getClassURIs()
        individuals = projection.getIndividualURIs()
        entities = classes.union(individuals)
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'entities.txt'), 'a') as f:
            for e in entities:
                f.write('%s\n' % e)

        # Extract and save axioms in Manchester Syntax
        logging.info('... Extract axioms ...')
        projection.createManchesterSyntaxAxioms()
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt'), 'a') as f:
            for ax in projection.axioms_manchester:
                axiom_sentence = [item for item in ax.split()]
                axiom_sentences.append(axiom_sentence)
                f.write('%s\n' % ax)
        logging.info('... %d axioms ...' % len(axiom_sentences))

        # Read annotations including rdfs:label and other literals from the ontology
        #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
        #                        2) None label annotations as sentences of the literal document
        logging.info('... Extract annotations ...')
        projection.indexAnnotations()
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'annotations.txt'), 'a') as f:
            for e in entities:
                if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                    label = list(projection.entityToPreferredLabels[e])[0]
                    v = pre_process_words(words=label.split())
                    uri_label[e] = v
                    f.write('%s preferred_label %s\n' % (e, v))
            for e in entities:
                if e in projection.entityToAllLexicalLabels:
                    for v in projection.entityToAllLexicalLabels[e]:
                        if (v is not None) and \
                            (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[
                                e])):
                            annotation = [e] + v.split()
                            annotations.append(annotation)
                            f.write('%s\n' % ' '.join(annotation))

        # project ontology to RDF graph (optionally) and extract walks
        if 'ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes':
            logging.info('... Calculate the ontology projection ...')
            projection.extractProjection()
            onto_projection_file = os.path.join(config['DOCUMENT']['cache_dir'], 'projection.ttl')
            projection.saveProjectionGraph(onto_projection_file)
            ONTO_FILE = onto_projection_file
        logging.info('... Generate walks ...')
        walks_ = get_rdf2vec_walks(onto_file=ONTO_FILE, walker_type=config['DOCUMENT']['walker'],
                                   walk_depth=int(config['DOCUMENT']['walk_depth']), classes=entities)
        logging.info('... %d walks for %d seed entities ...' % (len(walks_), len(entities)))
        walk_sentences += [list(map(str, x)) for x in walks_]

    # collect URI documents
    # two parts: axiom sentences + walk sentences
    URI_Doc = list()
    if 'URI_Doc' in config['DOCUMENT'] and config['DOCUMENT']['URI_Doc'] == 'yes':
        logging.info('Extracted %d axiom sentences' % len(axiom_sentences))
        URI_Doc = walk_sentences + axiom_sentences

    # Some entities have English labels
    # Keep the name of built-in properties (those starting with http://www.w3.org)
    # Some entities have no labels, then use the words in their URI name
    def label_item(item):
        if item in uri_label:
            return uri_label[item]
        elif item.startswith('http://www.w3.org'):
            return [item.split('#')[1].lower()]
        elif item.startswith('http://'):
            return URI_parse(uri=item)
        else:
            # return [item.lower()]
            return ''

    # read literal document
    # two parts: literals in the annotations (subject's label + literal words)
    #            replacing walk/axiom sentences by words in their labels
    Lit_Doc = list()
    if 'Lit_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Lit_Doc'] == 'yes':
        logging.info('\n\nGenerate literal document')
        for annotation in annotations:
            processed_words = pre_process_words(annotation[1:])
            if len(processed_words) > 0:
                Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
        logging.info('... Extracted %d annotation sentences ...' % len(Lit_Doc))

        for sentence in walk_sentences + axiom_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

    # for each axiom/walk sentence, generate mixture sentence(s) by two strategies:
    #   all): for each entity, keep its entity URI, replace the others by label words
    #   random): randomly select one entity, keep its entity URI, replace the others by label words
    Mix_Doc = list()
    if 'Mix_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Mix_Doc'] == 'yes':
        logging.info('\n\nGenerate mixture document')
        for sentence in walk_sentences + axiom_sentences:
            if config['DOCUMENT']['Mix_Type'] == 'all':
                for index in range(len(sentence)):
                    mix_sentence = list()
                    for i, item in enumerate(sentence):
                        mix_sentence += [item] if i == index else label_item(item=item)
                    Mix_Doc.append(mix_sentence)
            elif config['DOCUMENT']['Mix_Type'] == 'random':
                random_index = random.randint(0, len(sentence) - 1)
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    mix_sentence += [item] if i == random_index else label_item(item=item)
                Mix_Doc.append(mix_sentence)

    logging.info('\n\nURI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
    all_doc = URI_Doc + Lit_Doc + Mix_Doc
    logging.info('Time for document construction: %s seconds' % (time.time() - start_time))
    random.shuffle(all_doc)

    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if 'pre_train_model' not in config['MODEL'] or not os.path.exists(config['MODEL']['pre_train_model']):
        logging.info('\n\nTrain the language model')
        model_ = gensim.models.Word2Vec(all_doc, size=int(config['MODEL']['embed_size']),
                                        window=int(config['MODEL']['window']),
                                        workers=multiprocessing.cpu_count(),
                                        sg=1, iter=int(config['MODEL']['iteration']),
                                        negative=int(config['MODEL']['negative']),
                                        min_count=int(config['MODEL']['min_count']), seed=int(config['MODEL']['seed']))
    else:
        logging.info('\n\nFine-tune the pre-trained language model')
        model_ = gensim.models.Word2Vec.load(config['MODEL']['pre_train_model'])
        if len(all_doc) > 0:
            model_.min_count = int(config['MODEL']['min_count'])
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(config['MODEL']['epoch']))

    logging.info('Time for learning the language model: %s seconds' % (time.time() - start_time))
    return model_


'''
Embedding of a single input ontology using BERT
'''
def __perform_bert_ontology_embedding(config):
    if not BERT_AVAILABLE:
        raise ImportError("BERT embeddings require torch and transformers libraries. Please install them with: pip install torch transformers")

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    start_time = time.time()
      # Get BERT model parameters from config
    embedding_model = config.get('MODEL', 'embedding_model', fallback='bert-base')
    bert_model_name = config.get('MODEL', 'bert_model_name', fallback='bert-base-uncased')
    bert_pooling_strategy = config.get('MODEL', 'bert_pooling_strategy', fallback='mean')
    embed_size = int(config.get('MODEL', 'embed_size', fallback='768'))  # BERT base has 768 dimensions by default
    
    logging.info(f'Using BERT model type: {embedding_model} with model name: {bert_model_name} and {bert_pooling_strategy} pooling strategy')
    
    # Initialize BERT model and tokenizer based on the selected model type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f'Using device: {device}')
    
    # Select the appropriate model based on the embedding_model parameter
    model = None
    tokenizer = None
    
    if embedding_model == 'bert':
        # Standard BERT model implementation
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name).to(device)
        model.eval()  # Set model to evaluation mode
    elif embedding_model == 'bert-large':
        # BERT-Large model implementation
        bert_model_name = 'bert-large-uncased'  # Override with BERT-Large
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name).to(device)
        model.eval()
    elif embedding_model == 'sbert':
        # Sentence BERT implementation
        model = SentenceTransformer(bert_model_name).to(device)
    elif embedding_model == 'sapbert':
        # SapBERT model implementation
        if not bert_model_name or bert_model_name == 'bert-base-uncased':
            bert_model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BertModel.from_pretrained(bert_model_name).to(device)
        model.eval()
    else:
        raise ValueError(f"Unsupported BERT model type: {embedding_model}")
    
    # The rest of this function follows a similar flow to __perform_ontology_embedding
    # but with BERT-specific processing for embeddings
    
    if ('ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes') or \
        'pre_entity_file' not in config['DOCUMENT'] or 'pre_axiom_file' not in config['DOCUMENT'] or \
        'pre_annotation_file' not in config['DOCUMENT']:
        logging.info('Access the ontology ...')

        tax_only = (config['DOCUMENT']['projection_only_taxonomy'] == "yes")
        
        projection = OntologyProjection(config['BASIC']['ontology_file'], reasoner=Reasoner.STRUCTURAL,
                                        only_taxonomy=tax_only,
                                        bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                        additional_preferred_labels_annotations=set(),
                                        additional_synonyms_annotations=set(),
                                        memory_reasoner='13351')
    else:
        projection = None

    # Ontology projection
    if 'ontology_projection' in config['DOCUMENT'] and config['DOCUMENT']['ontology_projection'] == 'yes':
        logging.info('Calculate the ontology projection ...')
        projection.extractProjection()
        onto_projection_file = os.path.join(config['DOCUMENT']['cache_dir'], 'projection.ttl')
        projection.saveProjectionGraph(onto_projection_file)
        ontology_file = onto_projection_file
    else:
        ontology_file = config['BASIC']['ontology_file']

    # Extract and save seed entities (classes and individuals)
    # Or read entities specified by the user
    if 'pre_entity_file' in config['DOCUMENT']:
        entities = [line.strip() for line in open(config['DOCUMENT']['pre_entity_file']).readlines()]
    else:
        logging.info('Extract classes and individuals ...')
        projection.extractEntityURIs()
        classes = projection.getClassURIs()
        individuals = projection.getIndividualURIs()
        entities = classes.union(individuals)
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'entities.txt'), 'w') as f:
            for e in entities:
                f.write('%s\n' % e)

    # Extract axioms in Manchester Syntax if it is not pre_axiom_file is not set
    if 'pre_axiom_file' not in config['DOCUMENT']:
        logging.info('Extract axioms ...')
        projection.createManchesterSyntaxAxioms()
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt'), 'w') as f:
            for ax in projection.axioms_manchester:
                f.write('%s\n' % ax)

    # If pre_annotation_file is set, directly read annotations
    # else, read annotations including rdfs:label and other literals from the ontology
    #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
    #                        2) None label annotations as sentences of the literal document
    uri_label, uri_to_labels, annotations = dict(), dict(), list()

    if 'pre_annotation_file' in config['DOCUMENT']:
        with open(config['DOCUMENT']['pre_annotation_file']) as f:
            for line in f.readlines():
                tmp = line.strip().split()
                if tmp[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                    uri_label[tmp[0]] = pre_process_words(tmp[2:])
                else:
                    annotations.append([tmp[0]] + tmp[2:])

    else:
        logging.info('Extract annotations ...')
        projection.indexAnnotations()
        for e in entities:
            if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                label = list(projection.entityToPreferredLabels[e])[0]
                #Keeps only one
                uri_label[e] = pre_process_words(words=label.split())



                ##Populates dictionary with all labels per entity
                for label in projection.getPreferredLabelsForEntity(e):
                    #print("Preferred: " + label)
                    if e not in uri_to_labels:
                        uri_to_labels[e]=set()
                    #We add a list of words in the set
                    #print(pre_process_words(words=label.split()))
                    #print(uri_to_labels[e])
                    uri_to_labels[e].add(tuple(pre_process_words(words=label.split())))
		
                if e in projection.entityToSynonyms and len(projection.entityToSynonyms[e]) > 0:
                    for label in projection.getSynonymLabelsForEntity(e):
                        #print("Syn: " + label)
                        if e not in uri_to_labels:
                            uri_to_labels[e]=set()
                        #We add a list of words in the set
                        uri_to_labels[e].add(tuple(pre_process_words(words=label.split())))
                    
        for e in entities:
            if e in projection.entityToAllLexicalLabels:
                for v in projection.entityToAllLexicalLabels[e]:
                    if (v is not None) and \
                        (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[e])):
                        annotation = [e] + v.split()
                        annotations.append(annotation)

        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'annotations.txt'), 'w') as f:
            for e in projection.entityToPreferredLabels:
                for v in projection.entityToPreferredLabels[e]:
                    f.write('%s preferred_label %s\n' % (e, v))
            for a in annotations:
                f.write('%s\n' % ' '.join(a))

    # read URI document
    # two parts: walks, axioms (if the axiom file exists)
    walk_sentences, axiom_sentences, URI_Doc = list(), list(), list()
    if 'URI_Doc' in config['DOCUMENT'] and config['DOCUMENT']['URI_Doc'] == 'yes':
        logging.info('Generate URI document ...')
        walks_ = get_rdf2vec_walks(onto_file=ontology_file, walker_type=config['DOCUMENT']['walker'],
                                   walk_depth=int(config['DOCUMENT']['walk_depth']), classes=entities)
        logging.info('Extracted %d walks for %d seed entities' % (len(walks_), len(entities)))
        walk_sentences += [list(map(str, x)) for x in walks_]

        axiom_file = os.path.join(config['DOCUMENT']['cache_dir'], 'axioms.txt')
        if os.path.exists(axiom_file):
            for line in open(axiom_file).readlines():
                axiom_sentence = [item for item in line.strip().split()]
                axiom_sentences.append(axiom_sentence)
        logging.info('Extracted %d axiom sentences' % len(axiom_sentences))
        URI_Doc = walk_sentences + axiom_sentences

    # Some entities have English labels
    # Keep the name of built-in properties (those starting with http://www.w3.org)
    # Some entities have no labels, then use the words in their URI name
    def label_item(item):
        if item in uri_label:
            return uri_label[item]
        elif item.startswith('http://www.w3.org'):
            return [item.split('#')[1].lower()]
        elif item.startswith('http://'):
            return URI_parse(uri=item)
        else:
            # return [item.lower()]
            return ''

    # Create literal documents
    Lit_Doc = list()
    if 'Lit_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Lit_Doc'] == 'yes':
        logging.info('Generate literal document ...')
        for annotation in annotations:
            processed_words = pre_process_words(annotation[1:])
            if len(processed_words) > 0:
                Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
        
        for sentence in walk_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)
        
        for sentence in axiom_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

    # Create mixed documents
    Mix_Doc = list()
    if 'Mix_Doc' in config['DOCUMENT'] and config['DOCUMENT']['Mix_Doc'] == 'yes':
        logging.info('Generate mixture document ...')
        for sentence in walk_sentences + axiom_sentences:
            if config['DOCUMENT']['Mix_Type'] == 'all':
                for index in range(len(sentence)):
                    mix_sentence = list()
                    for i, item in enumerate(sentence):
                        mix_sentence += [item] if i == index else label_item(item=item)
                    Mix_Doc.append(mix_sentence)
            elif config['DOCUMENT']['Mix_Type'] == 'random':
                random_index = random.randint(0, len(sentence) - 1)
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    mix_sentence += [item] if i == random_index else label_item(item=item)
                Mix_Doc.append(mix_sentence)

    logging.info('URI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
    all_doc = URI_Doc + Lit_Doc + Mix_Doc
    logging.info('Time for document construction: %s seconds' % (time.time() - start_time))
    random.shuffle(all_doc)

    # Optional: save documents
    if config['DOCUMENT']['save_document'] == 'yes':
        with open(os.path.join(config['DOCUMENT']['cache_dir'], 'document_sentences.txt'), 'w') as f:
            for sentence in all_doc:
                for w in sentence:
                    f.write('%s ' % w)
                f.write('\n')
      # BERT-specific processing: Create embeddings for each entity/document
    logging.info(f'Creating {embedding_model.upper()} embeddings...')
    bert_start_time = time.time()
      # Create a custom embedding class that mimics gensim's Word2Vec interface
    class BertEmbedding:
        def __init__(self):
            self.wv = self  # For compatibility with Word2Vec interface
            # Set vector size based on model type
            if embedding_model == 'bert':
                self.vector_size = 768  # BERT base has 768 dimensions
            elif embedding_model == 'bert-large':
                self.vector_size = 1024  # BERT large has 1024 dimensions
            elif embedding_model == 'sbert':
                # SBERT models can have different dimensions, get it from the model
                if hasattr(model, 'get_sentence_embedding_dimension'):
                    self.vector_size = model.get_sentence_embedding_dimension()
                else:
                    self.vector_size = embed_size
            elif embedding_model == 'sapbert':
                self.vector_size = 768  # SapBERT typically has 768 dimensions
            else:
                self.vector_size = embed_size  # Use config value as fallback
                
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
    
    # Create the BERT embedding model
    bert_embeddings = BertEmbedding()
    
    # Process unique words/entities from all documents
    unique_tokens = set()
    for doc in all_doc:
        for token in doc:
            unique_tokens.add(token)
    
    logging.info(f'Found {len(unique_tokens)} unique tokens to embed')
      # Function to get BERT embedding for a token/sentence
    def get_bert_embedding(text):
        if embedding_model == 'bert' or embedding_model == 'bert-large':
            # Standard BERT embedding approach
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Get embeddings based on specified pooling strategy
            if bert_pooling_strategy == 'cls':
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif bert_pooling_strategy == 'mean':
                # Use mean of all token embeddings
                # Create attention mask to avoid padding tokens
                attention_mask = inputs['attention_mask']
                embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
                embedding = embedding.cpu().numpy()
            else:
                # Default to mean pooling
                attention_mask = inputs['attention_mask']
                embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
                embedding = embedding.cpu().numpy()
        
        elif embedding_model == 'sbert':
            # SentenceTransformer approach (SBERT)
            embedding = model.encode(text, convert_to_tensor=True).cpu().numpy()
        
        elif embedding_model == 'sapbert':
            # SapBERT approach
            tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, 
                                          padding='max_length', truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**tokens)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding[0]  # Return the embedding vector for this text
    
    # Batch process tokens to avoid memory issues
    batch_size = 32
    token_list = list(unique_tokens)
    
    for i in range(0, len(token_list), batch_size):
        batch_tokens = token_list[i:i+batch_size]
        for token in batch_tokens:
            # Process both URI tokens and literal tokens
            if token.startswith('http://'):
                # For URIs, try to use label if available, or URI components
                if token in uri_label:
                    text = ' '.join(uri_label[token])
                else:
                    text = ' '.join(URI_parse(uri=token))
            else:
                text = token
                
            # Get BERT embedding
            embedding = get_bert_embedding(text)
            
            # Store the embedding
            bert_embeddings.key_to_index[token] = len(bert_embeddings.index_to_key)
            bert_embeddings.index_to_key.append(token)
            bert_embeddings.vectors.append(embedding)
    
    # Convert vectors to numpy array for efficiency
    bert_embeddings.vectors = np.array(bert_embeddings.vectors)
    
    logging.info(f'Time for creating {embedding_model.upper()} embeddings: {time.time() - bert_start_time} seconds')
    
    return bert_embeddings

