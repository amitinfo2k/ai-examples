import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Artifact
from kfp.kubernetes import mount_pvc

# Component to parse tcpdump file (assumes input is a pcap file path)
#@component(base_image='amitinfo2k/pcap-pipeline:3.10',packages_to_install=['scapy'])
def parse_tcpdump(
    input_pcap: str,
    output_text: Output[Artifact]
):
    from scapy.all import rdpcap
    
    # Read the pcap file
    packets = rdpcap(input_pcap)
    
    # Generate text summaries for each packet
    summaries = [pkt.summary() for pkt in packets]
    
    # Write summaries to output artifact
    with open(output_text.path, 'w') as f:
        f.write('\n'.join(summaries))

# Component to embed packet summaries and store in ChromaDB
# Assumes ChromaDB service is deployed in the same Kubernetes cluster,
# accessible via service DNS (e.g., chroma-service in default namespace).
# Adjust the host/port as per your deployment.
@component(base_image='amitinfo2k/pcap-pipeline:3.10')
def embed_and_store(
    input_text: Input[Artifact],
    collection_name: str
):
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    # Connect to ChromaDB service with v2 API and explicit tenant/database
    client = chromadb.HttpClient(
        host='chroma-service.chromedb.svc.cluster.local',
        port=8000,
        ssl=False,
        headers={
            "x-chroma-client-version": "2.0.0",
            "x-chroma-request-source": "kubeflow-pipeline"
        },
        tenant="default_tenant",
        database="default_database"
    )
    
    # Get or create collection with v2 API
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        # Collection doesn't exist, create it
        collection = client.create_collection(name=collection_name)
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Read packet summaries
    with open(input_text.path, 'r') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    # Generate embeddings
    embeddings = model.encode(texts).tolist()
    
    # Add to ChromaDB
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f'pkt_{i}' for i in range(len(texts))]
    )

# Component for RAG: Retrieve from ChromaDB and generate analysis
# Uses Gemini as an example LLM for generation; replace with your preferred LLM if needed.
# Set GEMINI_API_KEY as a secret in Kubeflow or via environment variables.
@component(
    base_image='amitinfo2k/pcap-pipeline:3.10',
    packages_to_install=['google-generativeai']
)
def rag_analysis(
    collection_name: str,
    query: str,
    output_analysis: Output[Artifact]
):
    import chromadb
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    import os
    
    # Configure Gemini API key (assume it's set as env var or secret)
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Connect to ChromaDB service with v2 API and explicit tenant/database
    client = chromadb.HttpClient(
        host='chroma-service.chromedb.svc.cluster.local',
        port=8000,
        ssl=False,
        headers={
            "x-chroma-client-version": "2.0.0",
            "x-chroma-request-source": "kubeflow-pipeline"
        },
        tenant="default_tenant",
        database="default_database"
    )

    # Get collection with v2 API
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        raise ValueError(f"Collection '{collection_name}' does not exist")
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed query
    query_embedding = model.encode(query).tolist()
    
    # Retrieve top relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # Build context from retrieved documents
    context = '\n'.join(results['documents'][0])
    
    # RAG prompt
    prompt = f"Analyze the network traffic based on this context:\n{context}\n\nQuery: {query}\nAnalysis:"
    
    # Generate response using Gemini (placeholder model; adjust as needed, e.g., 'gemini-1.5-flash')
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.7
        )
    )
    
    # Write analysis to output
    with open(output_analysis.path, 'w') as f:
        f.write(response.text.strip())

# Define the Kubeflow Pipeline
#@dsl.pipeline(
#    name='RAG Pipeline for TCPDump Analysis',
#    description='A pipeline that processes tcpdump data, stores embeddings in ChromaDB, and performs RAG-based analysis.'
#)
def rag_tcpdump_pipeline(
    input_pcap: str = '/mnt/pcap/sample.pcap',  # Path in mounted volume
    collection_name: str = 'tcpdump_collection',
    analysis_query: str = 'Detect any anomalies in the network traffic',
    pvc_name: str = 'pcap-storage'  # Name of the PVC
):
    # Step 1: Parse tcpdump to text summaries
    parse_step = parse_tcpdump(input_pcap=input_pcap)
    
    # Mount the PVC to the step and set image pull policy
    from kfp.kubernetes import mount_pvc, set_image_pull_policy
    mount_pvc(
        task=parse_step,
        pvc_name=pvc_name,
        mount_path='/mnt/pcap'
    )
    set_image_pull_policy(parse_step, 'IfNotPresent')
  # Step 2: Embed and store in ChromaDB
    embed_step = embed_and_store(
        input_text=parse_step.output,
        collection_name=collection_name
    )
    set_image_pull_policy(embed_step, 'IfNotPresent')
    
    # Step 3: Perform RAG analysis
    analysis_step = rag_analysis(
        collection_name=collection_name,
        query=analysis_query
    ).after(embed_step)
    set_image_pull_policy(analysis_step, 'IfNotPresent')

# Local execution functions
def run_locally(pcap_path: str, query: str):
    import os
    import tempfile

    # Configuration
    config = {
        'input_pcap': pcap_path,
        'collection_name': 'tcpdump_collection',
        'query': query
    }

    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define paths
        output_text = os.path.join(temp_dir, 'packet_summaries.txt')
        
        # Step 1: Parse PCAP
        print("Step 1: Parsing PCAP file...")
        with open(output_text, 'w') as f:
            # Local version of parse_tcpdump logic
            from scapy.all import rdpcap
            packets = rdpcap(config['input_pcap'])
            summaries = [pkt.summary() for pkt in packets]
            f.write('\n'.join(summaries))
        print(f"Parsed {len(summaries)} packets to {output_text}")

        # Step 2: Embed and store
        print("\nStep 2: Creating embeddings...")
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        # Initialize ChromaDB client
        client = chromadb.Client()
        
        # Get or create collection
        try:
            collection = client.get_collection(name=config['collection_name'])
            print(f"Using existing collection: {config['collection_name']}")
        except ValueError:
            collection = client.create_collection(name=config['collection_name'])
            print(f"Created new collection: {config['collection_name']}")
        
        # Read packet summaries
        with open(output_text, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts).tolist()
        
        # Add to ChromaDB
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=[f'pkt_{i}' for i in range(len(texts))]
        )
        print(f"Added {len(texts)} documents to ChromaDB")

        # Step 3: Perform RAG analysis
        print("\nStep 3: Running RAG analysis...")
        # Simple local RAG implementation
        query_embedding = model.encode([config['query']]).tolist()[0]
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, len(texts))  # Get top 5 or all if less than 5
        )
        
        # Display results
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Query: {config['query']}")
        print("\nRelevant Packet Summaries:")
        for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            print(f"\n{i}. [Score: {1 - score:.3f}] {doc}")
        print("\n" + "="*50 + "\n")

# Entry point for local execution
if __name__ == '__main__':
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TCPDump analysis locally')
    parser.add_argument('--pcap', type=str, required=True,
                      help='Path to the PCAP file to analyze')
    parser.add_argument('--query', type=str, 
                      default='Detect any anomalies in the network traffic',
                      help='Query for RAG analysis')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pcap):
        print(f"Error: PCAP file not found: {args.pcap}")
        exit(1)
        
    run_locally(pcap_path=args.pcap, query=args.query)

# For Kubeflow pipeline compilation
if os.getenv('KUBEFLOW_RUN', '0') == '1':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=rag_tcpdump_pipeline,
        package_path='rag_tcpdump_pipeline.yaml'
    )
