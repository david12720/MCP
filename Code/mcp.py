import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

# Init model and ChromaDB
model = SentenceTransformer(r"C:\AI_Models\all-MiniLM-L6-v2")  #  "all-MiniLM-L6-v2")  # free + fast
#chroma_client = PersistentClient(path="./file_index_db")

chroma_client = chromadb.PersistentClient(
    path="./file_index_db",
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection("files")
collection.delete(where={"id": {"$ne": ""}})


# Step 1: Load file names (add .pdf, .docx content later)
def get_all_files(folder_path: str):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.pdf', '.txt', '.docx')):  # customize types
                full_path = os.path.join(root, file)
                file_list.append(full_path)
    return file_list

# Step 2: Index filenames with embedding
def index_files(folder_path: str):
    files = get_all_files(folder_path)
    print(f"📂 Found {len(files)} files. Indexing...")
    for idx, file_path in enumerate(files):
        file_name = Path(file_path).name
        embedding = model.encode(file_name).tolist()

        collection.add(
            documents=[file_name],
            embeddings=[embedding],
            metadatas=[{"path": file_path}],
            ids=[f"file-{idx}"]
        )

    print(f"✅ Indexed {len(files)} files into vector DB.")

# Run
if __name__ == "__main__":
    folder = input("Enter path to your files folder: ")
    index_files(folder.strip())
