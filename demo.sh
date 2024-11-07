#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

trap 'killall stern || true' INT EXIT

cat jobset.yaml
read ans
clear
less do.py
read ans
clear

kustomize build | kubectl apply -f -

echo "tailing logs"
echo "stern -l jobset.sigs.k8s.io/jobset-name=ace-jax-demo"

stern -l jobset.sigs.k8s.io/jobset-name=ace-jax-demo &
pid=$!

read ans
kill $pid

pod_name=$(kubectl get pod -l jobset.sigs.k8s.io/jobset-name=ace-jax-demo -o jsonpath="{.items[0].metadata.name}")
kubectl exec -it $pod_name -- bash

read ans

kustomize build | kubectl delete -f -
