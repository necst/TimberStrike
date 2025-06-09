#!/bin/bash
set -e

# Ensure the arguments are passed, otherwise default to 5 for clients and 30 for rounds
NUM_CLIENTS=${1:-5}
NUM_ROUNDS=${2:-30}

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

pushd ..

echo "Starting server"
python3 -m bagging.server --pool-size="$NUM_CLIENTS" --num-rounds="$NUM_ROUNDS" --num-clients-per-round="$NUM_CLIENTS" &
sleep 10  # Sleep for 10s to give the server enough time to start

for i in $(seq 0 $(("$NUM_CLIENTS" - 1))); do
    echo "Starting client $i"
    python3 -m bagging.client --partition-id="$i" --num-partitions="$NUM_CLIENTS" --partitioner-type=exponential &
done

# Enable CTRL+C to stop all background processes
# shellcheck disable=SC2064
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

echo "All processes completed"

# Kill process listening on port 8080 if any
if lsof -ti:8080 &> /dev/null; then
    echo "Killing process listening on port 8080"
    kill -9 "$(lsof -ti:8080)"
fi

popd
