import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, dimension=384): # all-MiniLM-L6-v2 dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension) # Inner product (Cosine sim for normalized vectors)
        self.metadata = [] # mapping from faiss id to (candidate_id, chunk_type, text)
        
    def add_chunks(self, candidate_id, chunks, embeddings):
        """
        Add candidate chunks to the FAISS index.
        Assumes embeddings are normalized.
        """
        if len(embeddings) == 0:
            return
            
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        for idx, chunk in enumerate(chunks):
            self.metadata.append({
                "candidate_id": candidate_id,
                "type": chunk["type"],
                "text": chunk["text"]
            })
            
    def search(self, query_embedding, k=50):
        """
        Search for top k matches.
        query_embedding must be a 1D numpy array.
        """
        q = np.array([query_embedding])
        faiss.normalize_L2(q)
        
        distances, indices = self.index.search(q, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "candidate_id": meta["candidate_id"],
                "type": meta["type"],
                "text": meta["text"],
                "score": float(distances[0][i])
            })
        return results
