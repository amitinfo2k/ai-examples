all: build
build:
	docker build -t amitinfo2k/pcap-pipeline:3.10 -f images/Dockerfile.pipe .
	docker build -t amitinfo2k/pcap-scapy:3.10 -f images/Dockerfile.scapy .	

	kind load docker-image amitinfo2k/pcap-pipeline:3.10
	