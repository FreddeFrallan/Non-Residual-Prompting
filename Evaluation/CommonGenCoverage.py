import spacy
from tqdm import tqdm
from lemminflect import getAllLemmas

nlp = spacy.load('en_core_web_sm')
nlp.disable_pipes(['parser', 'ner'])


def attemptToGetAllLemmas(wordString, spacyLemma=None):
    tupList = getAllLemmas(wordString).values()
    lemmaSet = set([x[0] for x in tupList])
    spacyLemma = {nlp(wordString)[0].lemma_ if spacyLemma is None else spacyLemma}
    lemmaSet = lemmaSet.union(spacyLemma)
    if len(lemmaSet) == 0:
        lemmaSet.add(wordString)
    return lemmaSet


# Taken by CommonGen: https://github.com/INK-USC/CommonGen/blob/master/evaluation/PivotScore/evaluate.py
# preds: list of strings (generated sentences)
# concept_sets: list of word lists [[str11, str12, ...], [str21, str22, ...], ...]
def coverageScore(preds, concept_sets):
    covs = []
    preds = [p.lower() for p in preds]
    preds = list(nlp.pipe(preds))
    for p, cs in tqdm(zip(preds, concept_sets), total=len(concept_sets)):

        lemmasCs = set()
        for concept in cs:
            concept = concept.lower()
            lemmaSet = attemptToGetAllLemmas(concept)
            lemmasCs = lemmasCs.union(lemmaSet)

        lemmasText = set()

        for token in p:
            lemmaSet = attemptToGetAllLemmas(token.text, spacyLemma=token.lemma_)
            lemmasText = lemmasText.union(lemmaSet)

        cov = len(lemmasText&lemmasCs)/len(lemmasCs)
        covs.append(cov)
    return sum(covs)/len(covs)


def originalCoverageScore(preds, concept_sets):
    covs = []

    for p, cs in tqdm(zip(preds,concept_sets), total=len(concept_sets)):
        cs = set(cs)
        lemmas = set()
        # for token in p:
        for token in nlp(p):
            lemmas.add(token.lemma_)
        cov = len(lemmas&cs)/len(cs)
        covs.append(cov)
    return sum(covs)/len(covs)
