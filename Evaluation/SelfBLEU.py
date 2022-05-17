import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from multiprocessing import Pool
import spacy
import numpy as np
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')


def calcBleu(reference, candidate, weights):
    return sentence_bleu(reference, candidate, weights=weights, smoothing_function=SmoothingFunction().method1)


def calculateSelfBLEU(texts, ngram=5):
    if len(texts) == 1:
        return 0

    spacyTexts = list(nlp.pipe(texts))
    textsSplits = np.array([[token.text for token in t] for t in spacyTexts], dtype=object)

    arange = np.arange(len(textsSplits))
    weights = tuple((1. / ngram for _ in range(ngram)))

    pool = Pool(os.cpu_count())
    bleus = list()
    for idx, candidate in enumerate(textsSplits):
        reference = textsSplits[arange != idx].tolist()
        bleus.append(pool.apply_async(calcBleu, args=(reference, candidate, weights)))

    for idx, b in tqdm(enumerate(bleus), total=len(bleus)):
        bleus[idx] = b.get()

    pool.close()
    pool.join()

    return np.mean(bleus)
