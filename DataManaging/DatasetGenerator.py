from DataManaging import PromptGenerator
import numpy as np

class DataFormatter:

    def __init__(self, tokenizer, maxSeqLen=64, maxPromptLen=32, minWords=1, maxWords=3,
                 promptTemplate='Include the words: '):
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
        self.maxPromptLen = maxPromptLen
        self.minWords, self.maxWords = minWords, maxWords
        self.promptTemplate = promptTemplate

        self.sampleDistrb = {k: 0 for k in range(5)}
        self.diffCounter = {k: 0 for k in range(20)}
        self.sampleCounter = 0

        self.totalSizeSaves = 0
        self.totalSizeErrors = 0

    def _formatSample(self, ids, pIDs, pos):
        attentionMask = np.zeros((self.maxSeqLen, self.maxPromptLen))
        contextLabelMask, wInclusionLabelMask = np.zeros(self.maxSeqLen), np.zeros(self.maxSeqLen)

        seqLen, promptLen = len(ids), len(pIDs)
        attentionMask[:promptLen] = 1
        contextLabelMask[:seqLen - 1] = 1
        for start, end in pos:
            contextLabelMask[start:end] = 0
            wInclusionLabelMask[start:end] = 1

        paddedPrompt = pIDs + [0] * (self.maxPromptLen - promptLen)
        paddedSeq = ids + [0] * (self.maxSeqLen - seqLen)

        return paddedSeq, paddedPrompt, attentionMask, contextLabelMask, wInclusionLabelMask

    def createPromptText(self, words, sentLen=None):
        return self.promptTemplate.format(str.join(', ', words))

    def createFittingPromptTexts(self, targetWords, sentLen=None, returnIDs=False):  # Is implemented in inheritance
        raise NotImplementedError

    def _createPromptIDs(self, words, sentLen):
        fullPrompt = self.createPromptText(words, sentLen)
        return self.tokenizer.encode(fullPrompt)

    def createDataExample(self, ids, words, pos, probs=None, sentLen=None):
        if (probs is None):
            hasProbs = False
            probs = [1] * len(words)
        else:
            hasProbs = True

        uniqueWordsAndPositions, uniqueWordsAndProbs = {}, {}
        for w, wPos, wProb in zip(words, pos, probs):
            if (w not in uniqueWordsAndPositions):
                uniqueWordsAndPositions[w] = []
                uniqueWordsAndProbs[w] = []

            uniqueWordsAndPositions[w].append(wPos)
            uniqueWordsAndProbs[w].append(wProb)

        numWords = len(uniqueWordsAndPositions)
        if (numWords < self.maxWords):  # Trying this out to completely balance the num words
            return False, None

        numCandidates = min(np.random.randint(self.minWords, self.maxWords + 1), numWords)
        if (hasProbs == False):
            candidates = range(numWords)
            selectedWordIDs = np.random.choice(candidates, numCandidates, replace=False)
            uniqueWordList = list(uniqueWordsAndPositions.keys())
            selectedWords = [uniqueWordList[i] for i in selectedWordIDs]
        else:
            wordsAndMinProbs = [(k, np.min(v)) for k, v in uniqueWordsAndProbs.items()]
            wordsAndMinProbs.sort(key=lambda x: x[-1])
            selectedWords = [w for w, p in wordsAndMinProbs[:numCandidates]]

        # print("Selected words:", selectedWords)
        selectedPositions = []
        for w in selectedWords:
            selectedPositions.extend(uniqueWordsAndPositions[w])

        promptIDs = self._createPromptIDs(selectedWords, sentLen)
        if (len(promptIDs) > self.maxPromptLen):
            return False, None

        return True, self._formatSample(ids, promptIDs, selectedPositions)

    def clearSizeSavesAndErrors(self):
        self.totalSizeErrors = 0
        self.totalSizeSaves = 0


class ExampleDataFormatter(DataFormatter):

    def __init__(self, tokenizer, maxSeqLen=64, maxPromptLen=80, minWords=1, maxWords=3,
                 promptTemplate='Include the words: {}\n'):
        super().__init__(tokenizer, maxSeqLen, maxPromptLen, minWords, maxWords, promptTemplate)
        self.promptTemplate = 'Include the words: apple, monkey, tree\n' \
                              'Jim climbed up the tree like a monkey in order to pick the apple.\n' \
                              'Include the words: salt\n' \
                              'The majority of harvested salt originate from Paris.\n' \
                              'Include the words: {}\n'


class ExampleDataFormatter3WordsMin(DataFormatter):

    def __init__(self, tokenizer, maxSeqLen=64, maxPromptLen=80, minWords=3, maxWords=5,
                 promptTemplate='Include the words: {}\n'):
        super().__init__(tokenizer, maxSeqLen, maxPromptLen, minWords, maxWords, promptTemplate)
        self.promptTemplate = 'Include the words: apple, monkey, tree\n' \
                              'Jim climbed up the tree like a monkey in order to pick the apple.\n' \
                              'Include the words: geneva salt, table, travel, \n' \
                              'For you to be able to put salt on your table, it has to travel from geneva.\n' \
                              'Include the words: {}\n'


class ThreeExampleDataFormatter3WordsMin(DataFormatter):

    def __init__(self, tokenizer, maxSeqLen=64, maxPromptLen=128, minWords=3, maxWords=5,
                 promptTemplate='Include the words: {}\n'):
        super().__init__(tokenizer, maxSeqLen, maxPromptLen, minWords, maxWords, promptTemplate)
        self.promptTemplate = '10 words and included words. budget, cuts, due\n' \
                              'RDC service was eliminated in 1981 due to budget cuts.\n' \
                              '7 words and included words. done, mexico, would\n' \
                              'I would have done it about Mexico.\n' \
                              '16 words and included words. river, fly, decide, sky, throw\n' \
                              'They decide to throw all of their anxiety into the river, and fly into the sky.\n' \
                              '{} words and included words. {}\n'

    def createFittingPromptText(self, targetWords, sentLen=None, returnIDs=False):
        txt = self.promptTemplate.format(sentLen, str.join(', ', targetWords))
        if(returnIDs):
            return txt, self.tokenizer.encode(txt)

        return txt


class RandomPromptsDataFormatter(DataFormatter):

    def __init__(self, tokenizer, maxSeqLen=64, maxPromptLen=128, minWords=3, maxWords=5,
                 promptTemplate='Include the words: {}\n', numExamplesPerPrompt=3, maxPromptAttempts=1000,
                 storePrompts=False):
        super().__init__(tokenizer, maxSeqLen, maxPromptLen, minWords, maxWords, promptTemplate)
        # examplesPath = "PromptExamples-10k-3to5words-Max25.pkl"
        examplesPath = "PromptExamples-50k-3to5words-Max25.pkl"
        self.promptTemplate = '*** Generating Random Prompts ***' \
                              ', Using file: {}'.format(examplesPath)

        self.maxPromptAttempts = maxPromptAttempts
        self.questionFormulator = PromptGenerator.loadMiniQuestionFormulator(numExamplesPerPrompt, examplesPath,
                                                                             maxPromptLen)
        self.storePrompts = storePrompts
        if (self.storePrompts):
            self.storedPrompts = []
            self.maxStoredPrompts = 2000

    def createPromptText(self, words, sentLen=None):
        return self.questionFormulator.createPromptText(words, sentLen)

    def _createPromptIDs(self, words, sentLen):
        # textPrompt = self.createPromptText(words, sentLen)
        _, ids = self.createFittingPromptText(words, sentLen, returnIDs=True)
        return ids

    def createFittingPromptText(self, targetWords, sentLen=None, returnIDs=False):
        txt, ids, isSaved, isError = self.questionFormulator.createFittingPromptText(targetWords, sentLen,
                                                                                     self.maxPromptAttempts)
        self.totalSizeSaves += 1 if (isSaved and isError is False) else 0
        self.totalSizeErrors += 1 if isError else 0

        if (self.storePrompts):
            examples, instruction = self.questionFormulator.recentExamples, self.questionFormulator.recentInstruction
            delimiter, seperator = self.questionFormulator.recentDelimiter, self.questionFormulator.recentSeparator
            self.storedPrompts.append(
                {'Examples': examples, 'Instruction': instruction, 'Delimiter': delimiter, 'Separator': seperator})
            self.storedPrompts = self.storedPrompts[:self.maxStoredPrompts]

        if (returnIDs):
            return txt, ids
        return txt
