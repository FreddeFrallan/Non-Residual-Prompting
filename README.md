<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Fine-grained controllable text generation via Non-Residual Prompting</h3>

  <p align="center">
    To get started run InferenceExample.py

  </p>
</p>

### Installing

**TODO**: Download model?

Install the required python packages:

`pip install -r /path/to/requirements.txt`

### Evaluation
To evaluate texts, use `./evaluate.sh` that takes an input file with a json list of texts as a positional argument.

*Note that the order of the texts within this list must correspond to the order of samples in the evaluation dataset.*

This file got some settings that can be configured:
- **DATASET**: `common_gen` or `c2gen`
- **PPL_MODEL_NAME**: The name of the model to use, the paper uses `gpt2-xl`
- **PPL_BATCH_SIZE**: With many texts and access to a large GPU, this can be increased for faster evaluation.
- **CONTEXT**: The custom context that was used when generating the texts. For example, when generating with no context (common_gen) the model may need an initial string to condition the generation on, like 'The'.
- **SENTENCE_LEVEL**: Leave as `--sentence_level` for sentence-level evaluation, or empty string for Free Text evaluation.

