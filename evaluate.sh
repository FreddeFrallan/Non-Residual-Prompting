#!/bin/bash

# Input file with generated texts (json list of strings)
# e.g.: ["My first generated text.", "The second one is always better!"]
INPUT_FILE=$1
echo "Input file: ${INPUT_FILE}"

DATASET="common_gen"
# DATASET="c2gen"

PPL_MODEL_NAME="gpt2-xl"
# PPL_MODEL_NAME="gpt2"

# Increase this for faster evaluation if you have access to a large GPU
PPL_BATCH_SIZE=8

# Set this to the custom context used when generating texts, most likely empty if c2gen
CONTEXT=""

# Set this to empty string for free text evaluation (32 tokens)
SENTENCE_LEVEL="--sentence_level"

cmd="python Evaluation/EvaluateMain.py
        --input_file $INPUT_FILE
        --bs $PPL_BATCH_SIZE
        --ppl_model_name $PPL_MODEL_NAME
        --dataset $DATASET
        --context '$CONTEXT'
        $SENTENCE_LEVEL"

echo $cmd
$cmd

