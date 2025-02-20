"""Embeddings generation and management for CV analysis."""
from langchain_google_vertexai import VertexAIEmbeddings
import numpy as np


class EmbeddingsHandler:
    def __init__(self, project_id):
        """Initialize embeddings handler with Google Cloud project."""
        self.embeddings_model = VertexAIEmbeddings(
            project=project_id,
            model_name="text-embedding-005"
        )

    def generate_embeddings(self, texts):
        """Generate embeddings for text chunks."""
        if not isinstance(texts, list):
            texts = [texts]

        embeddings = self.embeddings_model.embed_documents(texts)
        return list(zip(texts, embeddings))

    def get_embeddings(self):
        """Retrieve all embeddings from database."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, candidate_id, chunk_text, embedding 
                    FROM cv_embeddings 
                    ORDER BY id
                """)
                return cur.fetchall()

    @staticmethod
    def similarity_search(query_embedding, stored_embeddings, k=5):
        """Perform similarity search against stored embeddings."""
        similarities = []
        query_embedding = np.array(query_embedding)

        for idx, embedding in enumerate(stored_embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((idx, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
