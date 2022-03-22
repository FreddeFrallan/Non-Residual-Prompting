from DecodingStrategies import VocabularyManager
from transformers import generation_tf_utils
from lemminflect import getAllInflections
import tensorflow as tf
import transformers
import numpy as np
import copy

OLD_REP_PENALTY_FUNC = generation_tf_utils._create_next_token_logits_penalties
TOKENIZER = transformers.AutoTokenizer.from_pretrained('gpt2')
VOCAB_MANAGER = VocabularyManager.VocabularyManager(TOKENIZER)
CURRENT_INFERENCE_CONTAINER = None
CURRENT_INCLUSION_WORDS = None
CURRENT_CONTEXT_LEN = None
CURRENT_SEQ_STATES = None
CURRENT_TARGET_LEN = None
CURRENT_NUM_BEAMS = None


class SequenceState:
    class WordState:
        class LemmaState:
            def __init__(self, ids):
                self.targetLen = len(ids)
                self.counter = 0
                self.ids = ids

                self.canBeIncluded = False
                self.wordType = VOCAB_MANAGER.getWordType(ids[0])
                self.isSpacedNewWord = self.wordType != VocabularyManager.WordTypes.CONTINUATION

            def getCurrentTargetToken(self):
                return self.ids[self.counter]

            def updateState(self, token):
                if (token == self.getCurrentTargetToken()):
                    self.counter += 1
                    if (self.counter == self.targetLen):
                        self.canBeIncluded = True
                        self.counter = 0
                        return True
                else:
                    self.counter = 0

                return False

            def hasStarted(self):
                return self.counter > 0

        def __init__(self, word):
            self.allIDs = getPossibleIDsForWord(word)
            self.lemmaStates = [self.LemmaState(ids) for ids in self.allIDs]
            self.canBeIncluded = False
            self.isIncluded = False

        def updateState(self, recentToken):
            for lemma in self.lemmaStates:
                if (lemma.updateState(recentToken)):
                    self.canBeIncluded = True

        def getBoostIDs(self, canIncludeWord=False):
            # If lemma started, only recommend continuation of that lemma
            if (self.hasStarted()):
                return [lemma.getCurrentTargetToken() for lemma in self.lemmaStates if lemma.hasStarted()]
            else:
                if (canIncludeWord and self.canBeIncluded):
                    return []

                if (canIncludeWord):
                    return [lemma.getCurrentTargetToken() for lemma in self.lemmaStates if lemma.isSpacedNewWord]
                else:
                    return [lemma.getCurrentTargetToken() for lemma in self.lemmaStates]

        def getBlockIDs(self):
            allIDs = []
            for lemma in self.lemmaStates:
                allIDs.extend(lemma.ids)
            return allIDs

        def checkIfIncluded(self, recentToken):
            if (VOCAB_MANAGER.getWordType(recentToken) != VocabularyManager.WordTypes.CONTINUATION):
                self.isIncluded = True
            self.canBeIncluded = False

        def hasStarted(self):
            return any([lemma.hasStarted() for lemma in self.lemmaStates])

    def __init__(self, targetWords, targetLan, sampleID, sequenceID, numBeams=1, inclusionFactor=5.5, maxBoost=0.25,
                 useBoost=False, useBlocks=False):
        self.wordStates = [self.WordState(w) for w in targetWords]
        self.targetLan = targetLan
        self.numBeams = numBeams
        self.sampleID = sampleID
        self.seqID = sequenceID
        self.generatedLen = 0
        self.finished = False

        # Currently caching does not support multiple beams!
        # assert numBeams == 1
        self.checkFunc = self.getPenalties
        # self.checkFunc = self.checkOnlyRecentSymbol if numBeams == 1 else self.checkFullSeq

        self.inclusionFactor = inclusionFactor
        self.maxBoost = maxBoost
        self.useBoosts = useBoost
        self.useBlocks = useBlocks

    def checkIfFinished(self, recentToken):
        if (self.finished):
            CURRENT_INFERENCE_CONTAINER.promptBlockMask[self.seqID] = 0
            return

        for state in self.wordStates:
            if (state.canBeIncluded):
                state.checkIfIncluded(recentToken)

        for state in self.wordStates:
            if (state.isIncluded == False):
                state.updateState(recentToken)

        # allIncluded = all([state.isIncluded or state.canBeIncluded for state in self.wordStates])
        allIncluded = all([state.isIncluded for state in self.wordStates])
        # if (recentToken == 13 or allIncluded):
        if (allIncluded):
            CURRENT_INFERENCE_CONTAINER.promptBlockMask[self.seqID] = 0
            self.finished = True
        else:
            CURRENT_INFERENCE_CONTAINER.promptBlockMask[self.seqID] = 1

    def getPenalties(self, seqIDs):
        if (self.finished):
            return [], []

        # for state in self.wordStates:
        #     if (state.canBeIncluded):
        #         state.checkIfIncluded(seqIDs[-1])
        #
        # for state in self.wordStates:
        #     if (state.isIncluded == False):
        #         state.updateState(seqIDs[-1])

        hasStartedWords = any([state.hasStarted() for state in self.wordStates])
        hasCanBeIncluded = any([state.canBeIncluded for state in self.wordStates])

        penalties, boosts = [], []
        for state in self.wordStates:
            if (state.isIncluded and self.useBlocks):
                penalties.extend(state.getBlockIDs())
            elif (self.useBoosts):
                if (hasStartedWords):
                    if (state.hasStarted()):
                        boosts.extend(state.getBoostIDs(hasCanBeIncluded))
                else:
                    boosts.extend(state.getBoostIDs(hasCanBeIncluded))
        return penalties, boosts

    def getCurrentlyIncludedWords(self, seqIDs):
        return self.checkFunc(seqIDs)

    def getBoostFactor(self, seqIDs):
        completionFactor = len(seqIDs) / self.targetLan
        boostFactor = np.exp(self.inclusionFactor * completionFactor) / np.exp(self.inclusionFactor)
        # print("Boost Factor:", len(seqIDs), boostFactor)
        return 1 + self.maxBoost * boostFactor


def getPossibleIDsForWord(word):
    wordLemmas = [word]
    for lemmas in getAllInflections(word).values():
        wordLemmas.extend(lemmas)
    wordLemmas = np.unique(wordLemmas)

    contexts = (('', 0), ('The ', 1))
    wordVersions = []
    for lemma in wordLemmas:
        wordVersions.extend([lemma.lower(), lemma.lower().capitalize(), lemma.upper()])

    ids = []
    for c, i in contexts:
        for w in wordVersions:
            ids.append(TOKENIZER.encode(c + w)[i:])
    return ids


def HFRepPenalty(input_ids, logits, repetition_penalty, returnTF=False):
    # create logit penalties for already seen input_ids
    token_penalties = np.ones(generation_tf_utils.shape_list(logits))
    prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
    for i, prev_input_id in enumerate(prev_input_ids):
        logit_penalized = logits[i].numpy()[prev_input_id]
        logit_penalties = np.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[logit_penalized < 0] = repetition_penalty
        logit_penalties[logit_penalized > 0] = 1 / repetition_penalty
        np.put(token_penalties[i], prev_input_id, logit_penalties)
    if (returnTF):
        return tf.convert_to_tensor(token_penalties, dtype=tf.float32)
    return token_penalties


def getWordInclusionScores(seqIDs, seqState, penalyValue=-10):
    penaltiesIDs, boostsIDs = seqState.getCurrentlyIncludedWords(seqIDs)
    boostsIDs = {k: 1 for k in np.unique(boostsIDs)}
    penaltiesIDs = {k: 1 for k in np.unique(penaltiesIDs) if k not in boostsIDs}

    boostValue = seqState.getBoostFactor(seqIDs)
    penalties = [(ids, penalyValue) for ids in penaltiesIDs.keys()]
    boosts = [(ids, boostValue) for ids in boostsIDs.keys()]
    return penalties, boosts


def customRepPenaltyFunc(input_ids, logits, repetition_penalty):
    relevantIDs = input_ids[:, CURRENT_CONTEXT_LEN:]
    # print(CURRENT_CONTEXT_LEN, input_ids.shape, relevantIDs.shape)
    if (relevantIDs.shape[-1] <= 0):
        repScores = np.ones(logits.shape)
    else:
        repScores = HFRepPenalty(relevantIDs, logits, repetition_penalty)
        if (CURRENT_SEQ_STATES is not None):
            # for i, (ids, targetWords) in enumerate(zip(input_ids.numpy(), CURRENT_INCLUSION_WORDS)):
            for i, (ids, seqState) in enumerate(zip(relevantIDs.numpy(), CURRENT_SEQ_STATES)):
                penalties, boosts = getWordInclusionScores(ids, seqState)
                for index, value in penalties:
                    value = value if logits[i, index] > 0 else -value
                    repScores[i, index] = value

                for index, value in boosts:
                    value = 1 / value if logits[i, index] < 0 else value
                    repScores[i, index] = value

    return tf.convert_to_tensor(repScores, dtype=tf.float32)


def nextTokenHook(nextTokens):
    if (CURRENT_SEQ_STATES is not None):
        for token, seqState in zip(nextTokens.numpy(), CURRENT_SEQ_STATES):
            seqState.checkIfFinished(token)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def nextTokenBeamsHook(nextTokenBeams):
    global CURRENT_SEQ_STATES
    if (CURRENT_SEQ_STATES is not None):
        counter = 0
        newBeams = []
        for i, beamChunk in enumerate(chunks(nextTokenBeams, CURRENT_NUM_BEAMS)):
            for _, token, beamID in beamChunk:
                try:
                    # oldSeqID = CURRENT_SEQ_STATES[counter].seqID
                    beam = copy.deepcopy(CURRENT_SEQ_STATES[beamID])
                    beam.seqID = counter
                    if (isinstance(token, int)):
                        beam.checkIfFinished(token)
                    else:
                        beam.checkIfFinished(token.numpy())
                    newBeams.append(beam)
                except:
                    pass
                counter += 1

        CURRENT_SEQ_STATES = newBeams


def setCurrentInclusionWords(wordSets, sentLen, numBeams=1, inclusionFactor=5.5, maxInclusion=0.5, useBoost=False,
                             useBlocks=False, contextLen=0):
    global CURRENT_SEQ_STATES, CURRENT_NUM_BEAMS, CURRENT_CONTEXT_LEN
    if (wordSets is None):
        return

    CURRENT_NUM_BEAMS = numBeams
    CURRENT_CONTEXT_LEN = contextLen
    CURRENT_SEQ_STATES, seqCounter = [], 0
    for i, words in enumerate(wordSets):
        for _ in range(numBeams):
            CURRENT_SEQ_STATES.append(
                SequenceState(words, sentLen, i, seqCounter, numBeams, inclusionFactor, maxInclusion, useBoost=useBoost,
                              useBlocks=useBlocks))
            seqCounter += 1


print("Inserting custom RepPenalty Func")
generation_tf_utils._create_next_token_logits_penalties = customRepPenaltyFunc


def main():
    inclusionWords = [
        ['us', 'moon', 'television', 'russia'],
        ['china', 'soviet', 'news', 'people'],
    ]

    setCurrentInclusionWords(inclusionWords, 1, 1)
