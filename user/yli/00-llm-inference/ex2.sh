#!/bin/bash
set -exo pipefail

# model list
MODELS_TO_TEST="gemma3:4b-it-q4_K_M gemma3:4b-it-q8_0 gemma3:27b-it-q4_K_M"

# value list
echo "model_name,prompt_speed,generation_speed,combined_speed"

# run all models
for model in $MODELS_TO_TEST; do

    # >&2 - print in stderr, not in results.csv 
    echo "--- Processing: $model ---" >&2

    echo "Pulling $model (if not present)..." >&2
    # silent pull output
    ollama pull $model > /dev/null

    # get $OUTPUT and get performance metrics values
    echo "Running benchmark for $model..." >&2
    OUTPUT=$(ollama-benchmark --models $model --prompts "who is george p burdell?")

    PROMPT_SPEED=$(echo "$OUTPUT" | grep "Prompt Processing:" | awk '{print $3}')
    GEN_SPEED=$(echo "$OUTPUT" | grep "Generation Speed:" | awk '{print $3}')
    COMBINED_SPEED=$(echo "$OUTPUT" | grep "Combined Speed:" | awk '{print $3}')

    # print results in csv format
    echo "$model,$PROMPT_SPEED,$GEN_SPEED,$COMBINED_SPEED"

    echo "--- Finished: $model ---" >&2
done

echo "--- All benchmarks finished! ---" >&2