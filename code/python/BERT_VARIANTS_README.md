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

A test script is included (`test_owl2vec_bert_variants.py`) that validates each BERT variant.

To run the test:

```bash
python test_owl2vec_bert_variants.py
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
