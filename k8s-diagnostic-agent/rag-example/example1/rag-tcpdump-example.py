__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now, your other imports and code follow
import os
import re
from typing import List, Dict
import getpass


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. Set up your environment and API key ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

# --- 2. Data Ingestion & Processing ---
# A simplified function to simulate parsing a tcpdump file.
def parse_tcpdump_text(file_path: str) -> List[Dict]:
    """
    Parses a text file of tcpdump output and returns a list of dictionaries.
    Each dictionary represents a network event.
    """
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(
                r'(\d{2}:\d{2}:\d{2}.\d{6}) IP ([\d.]+).(\d+) > ([\d.]+).(\d+): (.+)',
                line
            )
            if match:
                timestamp, src_ip, src_port, dst_ip, dst_port, details = match.groups()
                events.append({
                    "timestamp": timestamp,
                    "source_ip": src_ip,
                    "source_port": src_port,
                    "destination_ip": dst_ip,
                    "destination_port": dst_port,
                    "protocol_details": details,
                    "full_line": line
                })
    return events

# Sample tcpdump data saved to a file
tcpdump_content = """
15:30:00.123456 IP 192.168.1.100.45678 > 104.26.2.228.80: Flags [S], seq 12345, win 65535, length 0
15:30:00.223456 IP 104.26.2.228.80 > 192.168.1.100.45678: Flags [S.], seq 67890, ack 12346, win 65535, length 0
15:30:00.323456 IP 192.168.1.100.45678 > 104.26.2.228.80: Flags [.], ack 67891, win 65535, length 0
15:30:00.423456 IP 192.168.1.100.45678 > 104.26.2.228.80: Flags [P.], seq 12346:12396, ack 67891, win 65535, length 50
15:30:01.567890 IP 172.16.2.5.34567 > 8.8.8.8.53: 2005+ A? google.com. (28)
15:30:01.667890 IP 8.8.8.8.53 > 172.16.2.5.34567: 2005 A 142.250.190.142 (44)
15:30:02.777777 IP 192.168.1.100.55555 > 20.30.40.50.443: Flags [S], seq 98765, win 65535, length 0
15:30:02.888888 IP 20.30.40.50.443 > 192.168.1.100.55555: Flags [S.], seq 12345, ack 98766, win 65535, length 0
15:30:02.999999 IP 192.168.1.100.55555 > 20.30.40.50.443: Flags [.], ack 12346, win 65535, length 0
"""

with open("tcpdump_data.txt", "w") as f:
    f.write(tcpdump_content)

# Process the data
parsed_events = parse_tcpdump_text("tcpdump_data.txt")
documents = [event["full_line"] for event in parsed_events]

# --- 3. Indexing (Embedding and Storing) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    # The `is_separator_regex_split` parameter is no longer supported and must be removed.
    # is_separator_regex_split=False  <-- This is the problematic line.
)
texts = text_splitter.create_documents(documents)

# Initialize the embedding model with a Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a vector database from the documents and embeddings
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings
)

print(f"Indexed {len(texts)} chunks of tcpdump data.")

# --- 4. Retrieval & Generation ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

rag_prompt_template = """
You are a network analysis assistant. Your role is to analyze and summarize network traffic logs provided as context.
You must answer the question based ONLY on the provided context. If the answer is not in the context, say "I could not find the answer in the provided network logs."

Context:
{context}

Question:
{question}
"""

rag_prompt = PromptTemplate.from_template(rag_prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# --- 5. Ask Questions ---
questions = [
    "What are the details of the TCP connection between 192.168.1.100 and 104.26.2.228?",
    "Were there any DNS queries made in the logs?",
    "What is the traffic related to IP address 20.30.40.50?",
    "Were there any connections to a suspicious IP address starting with 10.?"
]

for q in questions:
    print(f"\n--- Question: {q} ---")
    response = qa_chain.invoke({"query": q})
    print("Answer:")
    print(response["result"])
    print("--- Retrieved Sources: ---")
    for doc in response["source_documents"]:
        print(doc.page_content)
