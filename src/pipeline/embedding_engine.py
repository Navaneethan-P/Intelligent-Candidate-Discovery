import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts, batch_size=32):
        if not texts:
            return np.array([])
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
