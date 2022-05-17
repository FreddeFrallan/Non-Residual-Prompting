import math

import numpy as np
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from CommonGenCoverage import coverageScore
from SelfBLEU import calculateSelfBLEU
from VanillaGPT2 import VanillaGPT2

nlp = spacy.load('en_core_web_sm')


def splitToFirstSentences(generatedTexts):
    sentences = []
    spacyTexts = nlp.pipe(generatedTexts)
    for text in spacyTexts:
        if len(text.text) > 0:
            s = next(text.sents).text
        else:
            s = ""
        sentences.append(s.strip("\n").split("\n")[0])
    return sentences


# perplexityModel should be an instance of VanillaGPT2
# def benchCommonGenPerplexity(contextIds, generatedIds, perplexityModel, batchSize):
def benchCommonGenPerplexity(contexts, generated, perplexityModel, batchSize):
    numSamples = len(generated)
    perplexities = []
    print("Measuring perplexity...")
    # zipped = list(zip(contextIds, generatedIds))
    for i in tqdm(range(0, numSamples, batchSize)):
        # perplexities += perplexityModel.perplexity(preds[i:i + batchSize])
        perplexities += perplexityModel.perplexityFromContextAndGenerated(contexts[i:i + batchSize],
                                                                          generated[i:i + batchSize])

    for i in range(numSamples - 1, -1, -1):
        if math.isnan(perplexities[i]):
            del perplexities[i]
            print("Removed nan value from perplexities!, idx:", i)

    # import json
    # with open("ppls.json", 'w') as f:
        # pythPpls = [float(p) for p in perplexities]
        # json.dump(pythPpls, f)
        # print("Wrote ppls!")

    meanPerplexity = float(np.mean(perplexities))
    return meanPerplexity


def benchMetrics(generatedTextsNoContext, contexts, pplModel, batchSize, conceptSets,
                 doCov=True, doOrgCov=True, doSB=True, doPPL=True, doSemCov=True, sentenceLevel=False):
    if sentenceLevel:
        generatedTextsNoContext = splitToFirstSentences(generatedTextsNoContext)

    numEmpty = 0
    for g in generatedTextsNoContext:
        if len(g) == 0:
            numEmpty += 1
    print("Num empty samples:", numEmpty)
    coverage, coverageOrg, meanPerplexity, bleu5, semCoverage, avgNumTokens = [-1] * 6

    if isinstance(pplModel, str):
        tokenizer = AutoTokenizer.from_pretrained(pplModel)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = pplModel.tokenizer

    print("Calculating average #tokens")
    lens = [len(tokenizer.encode(t)) for t in generatedTextsNoContext]
    avgNumTokens = float(np.mean(lens))
    avgNumTokens = "%.4f" % round(avgNumTokens, 4)
    print("Average #tokens:", avgNumTokens)

    if doCov:
        print("Measuring coverage...")
        coverage = coverageScore(generatedTextsNoContext, conceptSets)
        coverage = "%.4f" % round(coverage, 4)
        print("Coverage:", coverage)

    if doSB:
        print("Calculating Self-BLEU...")
        bleu5 = calculateSelfBLEU(generatedTextsNoContext, ngram=5)
        bleu5 = "%.4f" % round(bleu5, 4)
        print("Self-BLEU:", bleu5)

    if doPPL:
        if isinstance(pplModel, str):
            print("Loading perplexity model:", pplModel)
            pplModel = VanillaGPT2(pplModel)

        meanPerplexity = benchCommonGenPerplexity(contexts, generatedTextsNoContext, pplModel, batchSize)
        meanPerplexity = "%.4f" % round(meanPerplexity, 4)
        print("Mean Perplexity:", meanPerplexity)

    return coverage, coverageOrg, meanPerplexity, bleu5, semCoverage


if __name__ == '__main__':
    """
    texts = ["Harry Potter really liked Dobby",
             "really liked Dobby.",
             " really liked Dobby.",
             "Harry Potter really liked Dobby.",
             "Harry Potter really liked Dobby. Despite his ugly nature.",
             "Why on earth did Ron want to play chess all the time?! I mean, why!?",
             "This is amazing, said Hermione. Harry did not agree.",
             "This is amazing, said Hermione. Harry did not agree.",
             "This is amazing, said Hermione. Harry did not agree.",
             "This is amazing, said Hermione. Harry did not agree.",
             "This is amazing, said Hermione. Harry did not agree.",
             ] + ["Why do I bother with this? Lol"] * 100

    st = time.time()
    texts += [""]
    splits = splitToFirstSentences(texts)
    print("Time Elapsed:", time.time() - st)
    for s in splits[:5]:
        print(s)
    """

    exes = [
        "The dog is run over by a car and drowns on the beach.com website.",
        "The dough is then rolled into a piece of paper, pinched and cut.\".\"Roll the roll out again.„.‟"
    ]
    splits = splitToFirstSentences(exes)
    print(splits[0])
    print(splits[1])
