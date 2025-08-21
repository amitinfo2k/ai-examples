# install chrome DB


## Install chromeDB in k8s cluster
```
kubectl create namespace chromedb
kubectl create -f  chrome-db.yaml
kubectl get all -n chromedb
```

## Install dependancies

```
python3 -m venv .venv
source .venv/bin/activate
pip install chromadb sentence-transformers
```

## Port Forward

```
kubectl port-forward -n chromedb svc/chroma-service 8000:8000
```

## Run the example

```
python3 validate_chroma.py
```


