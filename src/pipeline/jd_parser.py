import google.generativeai as genai
import json
from config import GEMINI_API_KEY

# Use gemini-2.5-flash for JD parsing — faster and higher free-tier quota
JD_PARSER_MODEL = 'gemini-2.5-flash'

class JDParser:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(JD_PARSER_MODEL)

    def parse_jd(self, raw_jd):
        prompt = f"""
        You are an expert technical recruiter. Read the following job description and extract the key requirements into distinct dimensions.
        Return the result EXACTLY as a valid JSON object with string keys and string values.
        Do not include any markdown formatting like ```json ... ```, just the raw JSON object.
        
        The dimensions should be roughly categorized into:
        - "role_core": The core technical focus and main responsibilities — what the person will actually build.
        - "infrastructure": The specific databases, cloud, vector stores, or infrastructure tools required.
        - "evaluation": The metrics, testing, or evaluation frameworks mentioned (NDCG, MRR, A/B testing, etc).
        - "engineering": The general software engineering languages, system design, scaling, and code quality requirements.
        - "culture_fit": The cultural and behavioral attributes sought — startup DNA, shipping mentality, product thinking.

        Each dimension value should be a detailed sentence summarizing all relevant requirements from the JD for that dimension.
        If a category is not explicitly mentioned, infer the closest related requirements from the JD context.

        Job Description:
        {raw_jd}
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            dimensions = json.loads(text.strip())
            return dimensions
        except Exception as e:
            print(f"Error parsing JD via LLM: {e}")
            # Fallback to a default if the LLM fails
            return {
                "role_core": "Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with embeddings-based retrieval systems deployed to real users. Own the intelligence layer: ranking, retrieval, and matching systems.",
                "infrastructure": "Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Operational experience with embedding drift, index refresh, retrieval-quality regression.",
                "evaluation": "Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. Set up offline benchmarks, online A/B testing, recruiter-feedback loops.",
                "engineering": "Strong Python with emphasis on code quality. Background in distributed systems or large-scale inference optimization. Built scalable ranking, search, or recommendation system to real users at meaningful scale.",
                "culture_fit": "Scrappy product-engineering attitude. Willing to ship a working system quickly. Startup 0-to-1 experience. Comfortable wearing multiple hats. Plans to stay 3+ years."
            }
