import numpy as np
import torch
import nltk


class SentenceWordInclusionDataset(torch.utils.data.Dataset):

    def __init__(self, dataSamples, tokenizer, maxSeqLen=32, maxPromptLen=32, minWords=3, maxWords=6,
                 randomSampleBufferSize=10000, padLabelToken=-100):
        self.maxPromptLen = maxPromptLen
        self.dataSamples = dataSamples
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen

        self.randomSampleBufferSize = randomSampleBufferSize
        self.minWords = minWords
        self.maxWords = maxWords
        self.setSampleBuffer()

        self.padLabelToken = padLabelToken

    def setSampleBuffer(self):
        self.numSampleWordsBuffer = np.random.randint(self.minWords, self.maxWords, self.randomSampleBufferSize)

    def __len__(self):
        return len(self.dataSamples)

    def padData(self, ids, promptIDs):
        ids, promptIDs = ids[:self.maxSeqLen], promptIDs[:self.maxPromptLen]

        txtPadSize = self.maxSeqLen - len(ids)
        promptPadSize = self.maxPromptLen - len(promptIDs)

        labelMask = torch.tensor(ids + [self.padLabelToken] * txtPadSize)
        promptMask = [True] * len(promptIDs) + [False] * promptPadSize
        promptMask = torch.repeat_interleave(torch.tensor(promptMask).unsqueeze(0).unsqueeze(0), self.maxSeqLen, -2)

        return ids + [0] * txtPadSize, promptIDs + [0] * promptPadSize, promptMask, labelMask

    def generatePromptText(self, targetWords, sentLen):
        return "Write {} words and include: {}".format(sentLen, str.join(', ', targetWords))

    def getWordsFromText(self, text):
        allWords = nltk.word_tokenize(text)
        filteredWords = [w for w in allWords if str.isalpha(w)]
        return filteredWords

    def __getitem__(self, item):
        text = self.dataSamples[item]
        availableWords = self.getWordsFromText(text)
        uniqueWords = list(set([w.lower() for w in availableWords]))

        numSampledWords = min(self.numSampleWordsBuffer[item % self.randomSampleBufferSize], len(uniqueWords))
        if (numSampledWords == len(uniqueWords)):
            targetWords = uniqueWords
        else:
            targetWords = np.random.choice(uniqueWords, numSampledWords, replace=False)

        promptText = self.generatePromptText(targetWords, len(availableWords))
        ids, promptIDs = self.tokenizer.encode(text), self.tokenizer.encode(promptText)

        f = lambda x: torch.tensor(x)
        ids, promptIDs, promptMask, labelMask = self.padData(ids, promptIDs)
        return (f(ids), f(promptIDs), promptMask), f(labelMask)


def loadDummySingleSentenceDataset(tokenizer, batchSize, maxSeqLen=32, maxPromptLen=32):
    dummySentences = ['This is just a dummy sentence with some random words.'] * 1000
    dataset = SentenceWordInclusionDataset(dummySentences, tokenizer, maxSeqLen=maxSeqLen, maxPromptLen=maxPromptLen)
    return torch.utils.data.DataLoader(dataset, batch_size=batchSize)
