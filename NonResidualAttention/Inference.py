from NonResidualAttention import PromptModelSetup, InferenceContainers
from DataManaging import DatasetGenerator
from DecodingStrategies import WordInclusion
from nltk.tokenize import word_tokenize
import tensorflow as tf
import transformers


def HFSearchFunc(inferenceContainer, contextIDs, promptIDs, maxLen=64, numBeams=8):
    contextIDs = tf.convert_to_tensor(contextIDs, tf.int32)
    bSize, contextLen = tf.shape(contextIDs)
    inferenceContainer = InferenceContainers.HFDecodingMultiSentenceContainer(inferenceContainer, contextIDs, promptIDs,
                                                                              maxLen=maxLen,
                                                                              verbose=True, numBeams=numBeams)
    WordInclusion.CURRENT_INFERENCE_CONTAINER = inferenceContainer

    newMaxLen = int(contextLen.numpy() + maxLen)
    seqs = inferenceContainer.generate(contextIDs, max_length=newMaxLen, num_beams=numBeams,
                                       num_return_sequences=1,
                                       eos_token_id=None,
                                       pad_token_id=13, use_cache=False,
                                       repetition_penalty=1.25,
                                       )

    return [s.tolist() for s in seqs.numpy()]


def batchGeneration(inferenceContainer, tokenizer, contexts, promptIDs, maxSeqLen, numBeams):
    contextIDs = [tokenizer.encode(context) for context in contexts]
    seqs = HFSearchFunc(inferenceContainer, contextIDs, promptIDs, maxLen=maxSeqLen, numBeams=numBeams)
    for s in seqs:
        print(s)
    generatedTexts = [tokenizer.decode(s).strip() for context, s in zip(contexts, seqs)]
    return generatedTexts


def getValidPredictions(newTexts, words):
    validSeqs = []
    for i, txt in enumerate(newTexts):
        generatedWords = {w.lower().strip(): 1 for w in word_tokenize(txt)}
        if (all(w in generatedWords for w in words)):
            validSeqs.append(newTexts[i])

    print("Number of valid seqs: {}/{}".format(len(validSeqs), len(newTexts)))
    if (len(validSeqs) == 0):
        return newTexts
    else:
        return validSeqs


def main():
    base = 'gpt2-large'
    clmPath = 'PaperModel-GPT2Large-CLM'
    postWeights = 'PaperModel-GPT2Large-PostWeights'
    promptModelPath = 'PaperModel-GPT2Large-PromptModel'

    quadWords = [
        ['wikileaks', 'scandal', 'sweden', 'website'],
    ]

    targetWords = quadWords
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    dataFormatter = DatasetGenerator.RandomPromptsDataFormatter(tokenizer, numExamplesPerPrompt=3)

    numBeams = 4
    sentLen, seqLen = 15, 32
    context = 'The'

    # Create model and load weights using absolute path
    model = PromptModelSetup.PromptModelSetup(base, CLMWeightsPath=clmPath, promptModelWeightsPath=promptModelPath,
                                              postWeightsPath=postWeights)

    # # List of ints for the tokenized prompt for each sequence. Will be padded later, so does not have to be same length
    promptIDs = [dataFormatter._createPromptIDs(words, sentLen) for words in targetWords]
    contextLen = len(tokenizer.encode(context))
    WordInclusion.setCurrentInclusionWords(targetWords, sentLen, contextLen=contextLen, numBeams=numBeams)

    # The context string for each sequence, HAS TO BE SAME LENGTH
    # Will later be replaced with <StartText>
    sampleContexts = [context for _ in targetWords]
    result = batchGeneration(model, tokenizer, sampleContexts, promptIDs, seqLen, numBeams)
    for txt, words in zip(result, targetWords):
        print("------------", words)
        print(txt)
