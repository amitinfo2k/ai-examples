# Makefile
# Variables
DOCKER_REGISTRY ?= 
VERSION ?= latest
KUBE_NAMESPACE ?= jolt-platform

# Docker build targets
.PHONY: build
build: build-orchestrator build-validator build-generator build-frontend

build-orchestrator:
	docker build -t ${DOCKER_REGISTRY}jolt-orchestrator:${VERSION} -f docker/orchestrator/Dockerfile .

build-validator:
	docker build -t ${DOCKER_REGISTRY}jolt-validator:${VERSION} -f docker/validator/Dockerfile .

build-generator:
	docker build -t ${DOCKER_REGISTRY}jolt-generator:${VERSION} -f docker/generator/Dockerfile .

build-frontend:
	docker build -t ${DOCKER_REGISTRY}jolt-frontend:${VERSION} -f docker/frontend/Dockerfile .

# Docker push targets
push: push-orchestrator push-validator push-generator push-frontend

push-orchestrator:
	docker push ${DOCKER_REGISTRY}jolt-orchestrator:${VERSION}

push-validator:
	docker push ${DOCKER_REGISTRY}jolt-validator:${VERSION}

push-generator:
	docker push ${DOCKER_REGISTRY}jolt-generator:${VERSION}

push-frontend:
	docker push ${DOCKER_REGISTRY}jolt-frontend:${VERSION}


load:
	kind load docker-image jolt-orchestrator:latest --name kind
	kind load docker-image jolt-validator:latest --name kind
	kind load docker-image jolt-generator:latest --name kind
	kind load docker-image jolt-frontend:latest --name kind

# Kubernetes deploy targets
deploy:
	kubectl apply -f k8s/manifests/00-namespace.yaml
	kubectl apply -f k8s/manifests/01-configmap.yaml
	kubectl apply -f k8s/manifests/02-orchestrator.yaml
	kubectl apply -f k8s/manifests/03-validator.yaml
	kubectl apply -f k8s/manifests/04-generator.yaml
	kubectl apply -f k8s/manifests/05-frontend.yaml
	kubectl apply -f k8s/manifests/06-mcp-service.yaml
	kubectl apply -f k8s/manifests/07-secrets.yaml
redeploy: clean build load deploy

# Port-forwarding
port-forward-orchestrator:
	kubectl port-forward svc/jolt-orchestrator-service 8088:80 -n ${KUBE_NAMESPACE} --address 0.0.0.0	

port-forward-frontend:
	kubectl port-forward svc/jolt-frontend-service 8501:80 -n ${KUBE_NAMESPACE} --address 0.0.0.0

# Cleanup
clean:
	kubectl delete -f k8s/manifests/ --recursive=true
