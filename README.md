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

**TODO**: Download model and fix a proper readme.

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

### Contextualized CommonGen Dataset (C<sup>2</sup>Gen)
In this paper we also introduce a new dataset that is based on the CommonGen dataset. In the CommonGen dataset
the objective of the task is to generate text that includes a given set of target words and that adheres
to common sense. These examples are however all formulated without any context, where we believe that many application 
areas need to take context into account. 

Therefore, to complement CommonGen, we provide an extended test set of CommonGen, called Contextualized CommonGen Dataset (C<sup>2</sup>Gen)
where a context is provided for each set of target words. The task is therefore reformulated to both
generate commonsensical text which includes the given words, and also have the generated text adhere
to the given context as shown on the Figure below.

<img src="Images/img.png" height="300" />


The dataset is uploaded on HuggingFace, so you can directly inspect the dataset here and incorporate it into your framework: [AI-Nordics/C2Gen](https://huggingface.co/datasets/AI-Nordics/C2Gen).