import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
import chromadb

def ingest_data(data_dir: str):
    """Loads tcpdump .txt files, chunks them, and ingests into ChromaDB."""
    print("--- Ingestion Component Started ---")

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )

    print(f"Loading documents from: {data_dir}")
    documents = loader.load()
    if not documents:
        raise ValueError("No documents were loaded. Check the data directory and file extensions.")

    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    print("Initializing embedding model...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Connecting to ChromaDB service")
    chroma_client = chromadb.HttpClient(
        host='chroma-service.chromedb.svc.cluster.local',
        port=8000,
        ssl=False,
        headers={"x-chroma-client-version": "2.0.0"}
    )

    collection_name = "tcpdump_logs"
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists. Deleting for a fresh start.")
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(name=collection_name)
    except ValueError:
        print(f"Creating new collection '{collection_name}'")
        collection = chroma_client.create_collection(name=collection_name)

    print(f"Ingesting {len(docs)} chunks into ChromaDB...")
    ids = [str(i) for i in range(len(docs))]
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    
    embeddings = embedding_function.embed_documents(texts)
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print("--- Ingestion Component Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the tcpdump data.')
    args = parser.parse_args()
    ingest_data(args.data_dir)

