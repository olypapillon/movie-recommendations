import numpy as np
import requests

EMBED_DIM = 768
EMBED_MODEL = 'nomic-embed-text'
OLLAMA_URL = 'http://localhost:11434/api/embeddings'

# Fields used to build a movie's textual representation, in display order.
REPRESENTATION_FIELDS = [
    ('type', 'Type'),
    ('title', 'Title'),
    ('director', 'Director'),
    ('cast', 'Cast'),
    ('release_year', 'Released'),
    ('listed_in', 'Genres'),
    ('description', 'Description'),
]


def create_textual_representation(row):
    """Build the textual representation for a movie row (dict-like).

    Fields that are blank or missing are omitted entirely.
    """
    lines = []
    for key, label in REPRESENTATION_FIELDS:
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        value = str(value).strip()
        if not value:
            continue
        lines.append(f'{label}: {value}')
    return '\n'.join(lines)


def embed(text):
    """Return the embedding for a piece of text as a float32 numpy array."""
    res = requests.post(OLLAMA_URL,
                        json={'model': EMBED_MODEL, 'prompt': text})
    res.raise_for_status()
    return np.array(res.json()['embedding'], dtype='float32')
