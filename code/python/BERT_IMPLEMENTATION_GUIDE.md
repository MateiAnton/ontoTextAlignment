# Implementation Guide for BERT Variants in OWL2Vec\*

This guide provides practical information on how to implement and use the BERT variants in your ontology embedding projects.

## Workflow Options

### Option 1: Full OWL2Vec\* Pipeline with BERT variants

Use this option when you need the full ontology processing pipeline with BERT embeddings:

1. Create a configuration file (e.g., `my_config.cfg`):

```ini
[BASIC]
ontology_file = /path/to/your/ontology.owl  # Use .owl format for best compatibility

[DOCUMENT]
cache_dir = ./cache/
URI_Doc = yes
Lit_Doc = yes
Mix_Doc = no
walker = random
walk_depth = 3
ontology_projection = no  # Set to 'no' if you encounter ontology loading issues

[MODEL]
embedding_model = bert  # Options: bert, bert-large, sbert, sapbert
bert_model_name = bert-base-uncased  # Change to your preferred model
bert_pooling_strategy = mean
```

2. Run OWL2Vec\* with your config:

```python
from owl2vec_star.owl2vec_star import extract_owl2vec_model

model = extract_owl2vec_model(
    ontology_file="./my_ontology.owl",
    config_file="./my_config.cfg",
    uri_doc=True,
    lit_doc=True,
    mix_doc=False
)

# Use the model
vector = model["http://example.org/ontology#MyClass"]
```

### Option 2: Standalone BERT Embeddings

If you encounter issues with ontology loading or just need the embedding functionality:

```python
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

# Choose your model
variant = "sbert"  # Options: bert, bert-large, sbert, sapbert
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
if variant == "bert":
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        embedding = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), 1) / torch.sum(attention_mask, 1, keepdim=True)
        return embedding.cpu().numpy()[0]

elif variant == "sbert":
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(device)

    def get_embedding(text):
        return model.encode(text, convert_to_tensor=True).cpu().numpy()

elif variant == "sapbert":
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    def get_embedding(text):
        tokens = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512,
                                      padding='max_length', truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding[0]

# Generate embeddings for a batch of terms
terms = ["http://example.org/ontology#Person", "http://example.org/ontology#Student"]
embeddings = {}

for term in terms:
    # Extract local name for better embedding
    if "#" in term:
        text = term.split("#")[-1]
    elif "/" in term:
        text = term.split("/")[-1]
    else:
        text = term

    embeddings[term] = get_embedding(text)

# Now you can use the embeddings
```

## Best Practices

1. **Choose the right BERT variant for your domain**:

   - **bert**: General purpose text embedding
   - **bert-large**: Higher dimensional embeddings for more complex tasks
   - **sbert**: Better for semantic similarity tasks
   - **sapbert**: Optimized for biomedical ontologies

2. **Handle ontology loading issues**:

   - Use OWL format (.owl) instead of TTL (.ttl)
   - Disable ontology projection by setting `ontology_projection = no`
   - If still having issues, use the standalone approach (Option 2)

3. **Preprocessing URIs**:

   - Extract meaningful terms from URIs for better embeddings (use local names)
   - Consider leveraging labels when available

4. **Performance optimization**:
   - Use batching for processing multiple terms
   - Use GPU when available (`cuda` device)
   - Cache embeddings to disk to avoid recomputation

## Example Use Cases

### Ontology Alignment

```python
# Calculate similarity between concepts from different ontologies
embedding_1 = get_embedding("Person")
embedding_2 = get_embedding("Human")

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]
print(f"Similarity between Person and Human: {similarity}")
```

### Ontology Completion

```python
# Find the most similar concept in the ontology
def find_most_similar(query_embedding, concept_embeddings, top_n=5):
    similarities = {}
    for concept, embedding in concept_embeddings.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities[concept] = similarity

    # Sort by similarity score
    sorted_concepts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_concepts[:top_n]

# Example usage
results = find_most_similar(get_embedding("Teacher"), embeddings, top_n=3)
```

## Troubleshooting

1. **Memory errors with large ontologies**:

   - Reduce batch size
   - Process ontology in chunks
   - Use smaller BERT models (e.g., SBERT with MiniLM)

2. **Incorrect embeddings for URIs**:

   - Ensure proper extraction of meaningful terms from URIs
   - Use ontology labels when available

3. **Slow embedding generation**:
   - Use GPU acceleration
   - Implement caching for embeddings
   - Consider using smaller models (SBERT is generally faster)
