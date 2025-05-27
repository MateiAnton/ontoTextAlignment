# Enhanced BERT Support for owl2vec_star

This enhancement adds support for additional BERT-based embedding models to the owl2vec_star library.

## New BERT Models

The following BERT models are now supported:

1. **BERT (bert)** - The original BERT model (both base and large)
2. **BERT-Large (bert-large)** - Specifically optimized to use BERT-Large for larger embedding dimensions
3. **Sentence BERT (sbert)** - Uses SentenceTransformer for better semantic sentence embeddings
4. **SapBERT (sapbert)** - Uses the SapBERT model from Cambridge, optimized for biomedical text

## How to Use

To use these models, set the appropriate configuration in your .cfg file:

```ini
[MODEL]
# Specify which embedding model to use
embedding_model = bert  # Options: word2vec, bert, bert-large, sbert, sapbert

# Model-specific parameters
bert_model_name = bert-base-uncased  # Model name from Hugging Face or SentenceTransformers
bert_pooling_strategy = mean  # Pooling strategy: 'mean' or 'cls'
```

### Model-Specific Notes

- **BERT**: Uses standard BERT from Hugging Face, with pooling options (mean or cls)
- **BERT-Large**: Uses bert-large-uncased model with 1024-dimensional embeddings
- **SBERT**: Uses SentenceTransformer models for better sentence-level representations
- **SapBERT**: Uses the Cambridge SapBERT model, specialized for biomedical domains

## Testing

Several test scripts are included to validate the BERT variants:

### 1. Test all variants with owl2vec_star pipeline
```bash
python test_owl2vec_bert_variants.py
```

This will test all BERT variants sequentially with the full owl2vec_star pipeline.

### 2. Test a single variant with owl2vec_star
```bash
python test_single_bert_variant.py [variant] [model_name]
```

Examples:
```bash
# Test standard BERT
python test_single_bert_variant.py bert bert-base-uncased

# Test BERT-Large
python test_single_bert_variant.py bert-large bert-large-uncased

# Test Sentence BERT
python test_single_bert_variant.py sbert paraphrase-MiniLM-L6-v2

# Test SapBERT
python test_single_bert_variant.py sapbert cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

### 3. Standalone BERT tests (bypassing ontology)

If there are issues with ontology loading, these standalone test scripts focus only on the BERT embedding functionality:

```bash
# Simple BERT test with sample sentences
python simple_bert_test.py [variant] [model_name]

# More comprehensive test with mock ontology data
python bert_standalone_test.py [variant] [model_name]
```

Examples:
```bash
# Test standalone BERT
python simple_bert_test.py bert bert-base-uncased

# Test standalone SBERT
python bert_standalone_test.py sbert paraphrase-MiniLM-L6-v2
```

## Dependencies

The following additional dependencies are required:

- torch
- transformers
- sentence-transformers

Install them with:

```bash
pip install torch transformers sentence-transformers
```

## Troubleshooting

If you encounter errors with the ontology loading (such as "OntologyAccess object has no attribute 'graph'"), try:

1. Use OWL format files instead of TTL: Use `.owl` files instead of `.ttl` files
2. Use the standalone test scripts to verify BERT functionality independently
3. Disable ontology projection by setting `ontology_projection = no` in the config
