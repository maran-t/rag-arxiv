# retreive.py
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from typing import List, Dict, Any
import os

load_dotenv()

client = OpenAI()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_QDRANT_URL = os.getenv("QDRANT_URL", "localhost:6333")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "arxiv_vector_store03")

vector_db = QdrantVectorStore.from_existing_collection(
    collection_name=_COLLECTION_NAME,
    url=_QDRANT_URL,
    api_key=_QDRANT_API_KEY,
    embedding=embeddings
)

def query_api(q: str, k: int = 3) -> Dict[str, Any]:
    """
    Programmatic replacement of the original interactive script.

    Args:
      query: user query string (will be stripped).
      k: number of nearest neighbors to retrieve from Qdrant (default 3).

    Returns:
      dict with keys:
        - answer: str (the model output)
        - context: str (the concatenated retrieved context)
        - sources: list of source metadata dicts
      or {"error": "..."} on failure.
    """
    try:
        query = (q or "").strip()
        if not query:
            return {"error": "empty query"}

        search_results = vector_db.similarity_search(
            query=query,
            k=k
        )

        context = "\n\n\n".join([
            f"Title and Abstract:\n{result.page_content}\nSource: {result.metadata['source']}" 
            for result in search_results
        ])

        SYSTEM_PROMPT = f"""
You are a helpful AI assistant. Use only the most relevant part of the context to answer the query. if the user prompt/query is not related to the below context, say "I could not find this in the provided context."

Rules:
- Use the information in the context.
- The retrieved context may contain multiple entries.
- Select the one that best matches the query and ignore unrelated ones.
- Do not merge unrelated entries.
- If no entry answers the query, reply: "I could not find this in the provided context."
- Also include the Problem, Consequence, Contribution, Findings if you find it through context, must include full abstract along with your answer.
- If possible include more resources from the web about the title/abstract, if and only if the query are relevant to the context below.

Context:
{context}

When answering:
- Use a friendly, conversational tone (add emoji like 👍, 🚀, ✅ where natural) & Organize explanations with headers, bullet points.  
"""

        chat_completion = client.chat.completions.create(
            model = "gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        answer = chat_completion.choices[0].message.content

        sources = []
        for r in search_results:
            sources.append({
                "source": r.metadata.get("source"),
                "title": r.metadata.get("titles"),
                "score": getattr(r, "score", None)
            })

        return {"answer": answer, "context": context, "sources": sources}
    except Exception as e:
        return {"error": str(e)}

