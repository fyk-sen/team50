#!/bin/bash

echo "Removing Previous Deployment..."
minikube stop
minikube delete

echo "Starting Minikube..."
minikube start --driver=docker

echo "Checking Minikube Status..."
kubectl get nodes

echo "Setting Docker Environment..."
eval $(minikube docker-env)

echo "Building Docker Images..."
cd application
docker build -t application .
cd ../processing
docker build -t processing .

echo "Checking Docker Images..."
docker images

echo "Applying Kubernetes Configurations..."
cd ..
kubectl apply -f shared-pv.yaml
kubectl apply -f shared-pvc.yaml

echo "Checking Persistent Volume & Persistent Volume Claims..."
kubectl get pv
kubectl get pvc

echo "Applying Kubernetes Configurations..."
kubectl apply -f application/application.yaml
kubectl apply -f processing/processing.yaml

echo "Checking Pods..."
kubectl get pods

echo "Getting Application Service URL..."
minikube service application-service --url