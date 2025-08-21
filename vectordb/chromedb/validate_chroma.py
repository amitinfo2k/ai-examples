import chromadb

# 1. Connect to the ChromaDB instance running in Kubernetes
# The port-forwarding in the previous step makes it available on localhost
client = chromadb.HttpClient(host='localhost', port=8000)

print("✅ Connected to ChromaDB!")

# 2. Create or get a collection to store our data
# This is like a table in a traditional database
collection_name = "tcpdump_patterns"
print(f"🔄 Getting or creating collection: '{collection_name}'...")
collection = client.get_or_create_collection(name=collection_name)
print("✅ Collection is ready.")

# 3. Add some sample documents related to tcpdump patterns
# ChromaDB will automatically handle embedding these for you
print("🔄 Adding documents to the collection...")
collection.add(
    documents=[
        "A TCP SYN flood is a denial-of-service attack where an attacker sends a succession of SYN requests to a target's system.",
        "Port scanning is a technique to identify open ports on a host. It often looks like one IP sending many packets to different ports on another IP.",
        "In tcpdump, the [S] flag indicates a SYN packet, which is used to initiate a TCP connection.",
        "An ICMP flood, or ping flood, overwhelms a target with ICMP Echo Request packets."
    ],
    ids=["doc1", "doc2", "doc3", "doc4"] # Each document needs a unique ID
)
print("✅ 4 documents added.")

# 4. Now, query the collection to see if it can find relevant information
query_text = "What does a high volume of SYN packets from one source indicate?"
print(f"\n❓ Querying with: '{query_text}'")

results = collection.query(
    query_texts=[query_text],
    n_results=2 # Ask for the top 2 most relevant results
)

print("\n🔍 Query Results:")
for i, doc in enumerate(results['documents'][0]):
    print(f"  {i+1}. {doc}")

# You can also inspect the distances (lower is more similar)
# print("\nDistances:", results['distances'])
