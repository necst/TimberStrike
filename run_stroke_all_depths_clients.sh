#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <num_clients>"
  exit 1
fi

NUM_CLIENTS=$1

for MAX_DEPTH in {3..8}; do
  ./run_clients.sh "$NUM_CLIENTS" "$MAX_DEPTH" "stroke"
done

