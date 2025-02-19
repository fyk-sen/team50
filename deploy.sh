#!/bin/bash

echo "Removing Previous Deployment..."
kubectl delete pods --all
kubectl delete deployments --all
kubectl delete services --all
kubectl delete pvc --all
kubectl delete pv --all

minikube stop
minikube delete

echo "Starting Minikube..."
minikube start --driver=docker
eval $(minikube docker-env)

echo "Building Docker Images..."
docker build -t model ./model
docker build -t application ./application
docker build -t processing ./processing
docker build -t inference ./inference

echo "Applying Kubernetes Configurations..."
kubectl apply -f shared-pv.yaml
kubectl apply -f shared-pvc.yaml
kubectl apply -f mlflow-pvc.yaml
kubectl apply -f model-pvc.yaml

kubectl apply -f application/application.yaml
kubectl apply -f processing/processing.yaml
kubectl apply -f inference/inference.yaml
kubectl apply -f model/model.yaml
kubectl apply -f mlflow.yaml

echo "Checking Pods Status..."
# Wait until all pods are Running or Completed
while true; do
    STATUS=$(kubectl get pods --no-headers | awk '{print $3}' | grep -vE 'Running|Completed')

    if [ -z "$STATUS" ]; then
        echo "All pods are Running!"
        break
    else
        echo "Waiting for pods to be ready..."
        sleep 5
    fi
done

# Open application and MLflow service in two different terminals

# Open Application Service
gnome-terminal -- bash -c "echo 'Opening Application Service...'; xdg-open \$(minikube service application-service --url); exec bash"

# Open MLflow Service
gnome-terminal -- bash -c "echo 'Opening MLflow Service...'; xdg-open \$(minikube service mlflow-service --url); exec bash"