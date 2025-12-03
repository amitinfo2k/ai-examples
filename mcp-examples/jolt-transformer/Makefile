# Makefile for MCP JOLT Transformer

# Variables
IMAGE_NAME := mcp-jolt-server
IMAGE_TAG ?= latest
CONTAINER_NAME := mcp-jolt-server
PORT := 8081
WORKDIR := /app

# Kind Cluster Name
KIND_CLUSTER_NAME ?= kind-calico-multus

.PHONY: build run stop clean test build-run kind-load deploy

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Load the image into Kind cluster
kind-load:
	@echo "Loading image into Kind cluster $(KIND_CLUSTER_NAME)..."
	kind load docker-image $(IMAGE_NAME):$(IMAGE_TAG) --name $(KIND_CLUSTER_NAME)

# Build and load into Kind
deploy: build kind-load

# Run the container in detached mode
run:
	@echo "Starting container..."
	@docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8081 \
		-v $(PWD)/resources/input.json:$(WORKDIR)/input.json \
		-v $(PWD)/resources/jolt_spec.json:$(WORKDIR)/jolt_spec.json \
		$(IMAGE_NAME):$(IMAGE_TAG)

# Stop and remove the running container
stop:
	@echo "Stopping and removing container..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true

# Clean up Docker resources
clean: stop
	@echo "Removing Docker image..."
	@docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true

# Run tests (add your test commands here)
test:
	@echo "Running tests..."
	# Add your test commands here

# Build and run in one command
build-run: build run

# Show help
help:
	@echo "Available targets:"
	@echo "  build     - Build the Docker image"
	@echo "  kind-load - Load the image into Kind cluster (default: kind-calico-multus)"
	@echo "  deploy    - Build and load into Kind"
	@echo "  run       - Run the container"
	@echo "  stop      - Stop the running container"
	@echo "  clean     - Remove the Docker image and container"
	@echo "  test      - Run tests"
	@echo "  build-run - Build and run in one command"
	@echo "  help      - Show this help message"

# Set the default target
.DEFAULT_GOAL := help
