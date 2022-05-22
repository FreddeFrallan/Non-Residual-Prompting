from NonResidualAttention import PromptModelSetup
from DataManagement import PromptGenerator
import transformers


def loadPaperGPT2LargeModel():
    maxSeqLen, maxPromptLen = 32, 128

    base = 'gpt2-large'
    clmBasePath = 'gpt2-large'
    promptModelBasePath = 'Non-Residual-Prompting/GPT2-Large'
    postWeightsPath = 'Non-Residual-Prompting/GPT2-Large-Post-Transformation'

    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    model = PromptModelSetup.PromptModelSetup(base, CLMWeightsPath=clmBasePath,
                                              promptModelWeightsPath=promptModelBasePath,
                                              postWeightsPath=postWeightsPath)

    promptGenerator = PromptGenerator.PromptGenerator(tokenizer, maxPromptLen)
    return model, tokenizer, maxSeqLen, promptGenerator


AVAILABLE_MODELS = {
    'gpt2-large': loadPaperGPT2LargeModel
}


def loadModel(modelName):
    return AVAILABLE_MODELS[modelName]()
