FROM python:3.13.5-slim

WORKDIR /app

COPY . .

RUN pip install -e .

ENTRYPOINT ["python", "-m", "pcap_analyzer.cli"]
