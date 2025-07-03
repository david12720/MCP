from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Load same model and database
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path="./file_index_db")
collection = chroma_client.get_or_create_collection("files")

# Ask user for query
query =  "איטליה"  #input("What are you looking for? ").strip()

# Convert query to embedding
query_vec = model.encode(query).tolist()

# Search in vector DB
results = collection.query(
    query_embeddings=[query_vec],
    n_results=1,
    include=['documents', 'metadatas']
)

# Print results
print("\n🔍 Top matches:")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"📄 {doc} — {meta['path']}")
