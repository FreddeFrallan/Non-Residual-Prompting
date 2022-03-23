import os

GPU = input("GPU:")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

from NonResidualAttention import InferenceUtils
from DecodingStrategies import WordInclusion
import Utils

if __name__ == '__main__':
    model, tokenizer, maxSeqLen, promptGenerator = Utils.loadPaperGPT2LargeModel()
    sentLen, seqLen, numBeams, context = 15, 32, 4, 'The'

    targetWords = [
        ['wikileaks', 'scandal', 'sweden', 'website'],
    ]

    # # List of ints for the tokenized prompt for each sequence. Will be padded later, so does not have to be same length
    promptIDs = [promptGenerator.createPromptIDs(words, sentLen) for words in targetWords]
    contextLen = len(tokenizer.encode(context))
    WordInclusion.setCurrentInclusionWords(targetWords, sentLen, contextLen=contextLen, numBeams=numBeams)

    # The context string for each sequence, HAS TO BE SAME LENGTH
    # Will later be replaced with <StartText>
    sampleContexts = [context for _ in targetWords]
    result = InferenceUtils.batchGeneration(model, tokenizer, sampleContexts, promptIDs, seqLen, numBeams)
    for txt, words in zip(result, targetWords):
        print("------------", words)
        print(txt)
