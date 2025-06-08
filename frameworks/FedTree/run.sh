#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <N> <dataset>"
    echo "  <N>: Number of parties"
    echo "  <dataset>: Name of the dataset ("$dataset", breast)"
    exit 1
fi

N=${1:-2}
dataset=${2:-"$dataset"}

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

cleanup() {
    echo "Terminating all processes..."
    pkill -P $$  # Kills all child processes of this script
    wait
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT

mkdir -p client_messages
mkdir -p server_messages
mkdir -p logs
rm -rf client_messages/*
rm -rf server_messages/*

privacy_budget=$(grep -oP 'privacy_budget=\K[0-9.]+' examples/"$dataset"/"$dataset"_horizontal_server.conf)
max_depth=$(grep -oP 'max_depth=\K[0-9]+' examples/"$dataset"/"$dataset"_horizontal_server.conf)
echo "Privacy budget: $privacy_budget"

  ./build/bin/FedTree-distributed-server examples/"$dataset"/"$dataset"_horizontal_server.conf &

for (( i=0; i<N; i++ ))
do
    privacy_tech=$(grep -oP 'privacy_tech=\K\w+' examples/"$dataset"/"$dataset"_horizontal_p${i}.conf)
    echo "Privacy tech: $privacy_tech"
    if [ "$privacy_tech" == "dp" ]; then
        mkdir -p logs/"$dataset"/logs_"$max_depth"_dp_"$privacy_budget"
        rm -rf logs/"$dataset"/logs_dp_"$max_depth"_"$privacy_budget"/party${i}.log
        ./build/bin/FedTree-distributed-party ./examples/"$dataset"/"$dataset"_horizontal_p${i}.conf $i > logs/"$dataset"/logs_"$max_depth"_dp_"$privacy_budget"/party${i}.log &
    else # none, so no privacy tech
        mkdir -p logs/"$dataset"/logs_"$N"_"$max_depth"
        rm -rf logs/"$dataset"/logs_"$N"_"$max_depth"/party${i}.log
        ./build/bin/FedTree-distributed-party ./examples/"$dataset"/"$dataset"_horizontal_p${i}.conf $i > logs/"$dataset"/logs_"$N"_"$max_depth"/party${i}.log &
    fi
done

wait