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