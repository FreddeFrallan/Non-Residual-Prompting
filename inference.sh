#!/bin/bash

# The words to be included in the text
TARGET_WORDS="wikileaks scandal sweden website"

# The context to continue generating from
CONTEXT="The"

# The target sentence length to instruct the model with
SENTENCE_LENGTH=15

# How many tokens in the resulting text
GENERATE_LENGTH=32

# Number of beams for beam search
NUM_BEAMS=4


cmd="python InferenceExample.py
      --target_words $TARGET_WORDS
      --context $CONTEXT
      --sentence_length $SENTENCE_LENGTH
      --generate_length $GENERATE_LENGTH
      --num_beams $NUM_BEAMS"

$cmd
