#!/bin/bash
#SBATCH -Jollama --account=paceship-dsgt_clef2026
#SBATCH --nodes=1 --partition=gpu-rtx6000 --gres=gpu:1
#SBATCH -t1:00:00 -qembers -ologs/ollama/%j.out
set -ex

function find_free_port() {
    # NOTE(anthony): this finds a free port dynamically. Often the GPUs are
    # found on the same node/rack, and the job fails because the port is already
    # allocated, or you end up running inference on a pre-existing service. I
    # figured this out the hard way.
    local port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    lsof -i:$port && { echo "Port $port in use, exiting"; exit 1; }
    echo $port
}

function wait_for_ollama_host() {
    # https://github.com/ollama/ollama/issues/3341#issuecomment-2360577772
    local host=${1?"host required"}
    curl --retry 5 --retry-connrefused --retry-delay 0 -sf ${host}
}

function wait_for_completion_api() {
    # a useful snippet to wait for the service to be ready
    # this will exit the script if the service does not come up in time
    local port=${1?"port number required"}
    local model=${2?"model name required"}
    echo "Waiting for API..."
    for i in {1..50}; do
        if curl -s -w "%{http_code}" -o /dev/null http://localhost:$port/v1/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$model\",\"prompt\":\"Hi\",\"max_tokens\":1,\"temperature\":0}" \
            | grep -q "200"; then
            echo "API ready"; return 0
        fi
        [ $i -eq 50 ] && { echo "API timeout"; exit 1; }
        echo "Attempt $i/50..."; sleep 10
    done
}

function main() {
# hostname is useful for figuring out which node to ssh into for debugging
hostname
nvidia-smi

# load the module
module load ollama

# run the model server, and expose it to the current node
# we use a custom port to avoid conflicts
# you can use 0.0.0.0 if you want to access it from other nodes
PORT=$(find_free_port)
export OLLAMA_HOST=localhost:$PORT

ollama serve &
SERVER_PID=$!

# cleanup on exit
trap "echo 'Killing server pid $SERVER_PID'; kill $SERVER_PID" EXIT

# wait for the ollama endpoint to be ready
wait_for_ollama_host $OLLAMA_HOST

echo "Comparing model sizes:"
ollama-benchmark --verbose --models gemma3:270m --prompts "write a haiku on embedded systems"
ollama-benchmark --verbose --models gemma3:4b --prompts "write a haiku on embedded systems"
ollama-benchmark --verbose --models gemma3:27b --prompts "write a haiku on embedded systems"

echo "Done"

}

main "$@"
