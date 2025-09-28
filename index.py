from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path


load_dotenv()

def clean_text(s: str) -> str:
    return " ".join(str(s).split()) 

csv_path = Path(__file__).parent / "arxiv_data.csv"
loader = CSVLoader(
    file_path=str(csv_path),
    encoding="utf-8",
    csv_args={"delimiter": ","},
    content_columns=["titles", "abstracts"],
    metadata_columns=["titles", "terms"]
)
docs = loader.load()

for d in docs:
    d.page_content = clean_text(d.page_content)
    d.page_content = f"Title: {d.metadata['titles']}\n\nAbstract: {d.page_content.split('abstracts:')[1].strip()}"

texts = docs


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="arxiv_vector_store03",
    host="localhost",
    port=6333,
    batch_size=128
)

print("Vector store for confidential data created and persisted successfully.")
