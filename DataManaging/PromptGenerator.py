import transformers
import numpy as np
import pickle


class QuestionFormulator():
    def __init__(self, examples, numExamplesPerPrompt=3, maxPromptSize=128, bufferSize=10000, tokenizer=None):
        self.examples = examples
        self.instructionFormulations = [
            'Include', 'Include the words', 'Sentence should include', 'Text should include', 'Sentence must include',
            'Text must include', 'Using the words', 'Sentence using', 'Text using', 'Use the words',
            'target words', 'Inclusion words', 'Words to include', 'Included words'
        ]
        self.delimiterSigns = [': ', ' ', '> ', '. ']
        self.seperatorSign = [', ', ' ']
        self.finalSigns = ['\n', '.\n', ':\n']

        self.SentLenFormulation = '{} words and '

        self.maxPromptSize = maxPromptSize
        self.numExamplesPerPrompt = numExamplesPerPrompt
        self.randomIterator = self.createBufferRandomSelector(bufferSize)
        self.tokenizer = tokenizer if tokenizer is not None else transformers.AutoTokenizer.from_pretrained('gpt2')


    def createBufferRandomSelector(self, bufferSize):
        while True:
            instructions = np.random.randint(0, len(self.instructionFormulations), bufferSize)
            delimiters = np.random.randint(0, len(self.delimiterSigns), bufferSize)
            seperators = np.random.randint(0, len(self.seperatorSign), bufferSize)
            finals = np.random.randint(0, len(self.finalSigns), bufferSize)

            examples = np.random.randint(0, len(self.examples), (bufferSize, self.numExamplesPerPrompt))

            for exes, i, d, s, f in zip(examples, instructions, delimiters, seperators, finals):
                if (len(np.unique(exes)) != len(exes)):  # Never include the same example twice in one prompt
                    continue

                exes = [self.examples[e] for e in exes]
                yield exes, self.instructionFormulations[i], self.delimiterSigns[d], self.seperatorSign[s], \
                      self.finalSigns[f]

    def createPromptText(self, targetWords, sentLen=None):
        targetWords = [w.lower() for w in targetWords]
        examples, instruction, delimiter, seperator, final = next(self.randomIterator)

        sampleTexts = []
        for example, words in examples:
            example = example.strip()
            if (sentLen is None):
                txt = instruction + delimiter + str.join(seperator, words) + final + example
            else:
                numWords = len(example.split(' '))
                lenForm = self.SentLenFormulation.format(numWords)
                txt = lenForm + instruction.lower() + delimiter + str.join(seperator, words) + final + example

            sampleTexts.append(txt)

        # Add the Target Words
        if (sentLen is None):
            targetText = instruction + delimiter + str.join(seperator, targetWords) + final
        else:
            lenForm = self.SentLenFormulation.format(sentLen)
            targetText = lenForm + instruction.lower() + delimiter + str.join(seperator, targetWords) + final
        sampleTexts.append(targetText)

        return str.join("\n", sampleTexts)

    def createFittingPromptText(self, targetWords, sentLen=None, maxAttempts=1000):
        isSaved, isError = False, False
        for _ in range(maxAttempts):
            txt = self.createPromptText(targetWords, sentLen)
            ids = self.tokenizer.encode(txt)
            if (len(ids) <= self.maxPromptSize):
                break
            isSaved = True
        else:
            isError = True

        return txt, ids, isSaved, isError


def loadMiniQuestionFormulator(numExamplesPerPrompt=3, examplesPath="PromptExamples-10k-3to5words-Max25.pkl",
                               maxPromptSize=128):
    with open(examplesPath, 'rb') as fp:
        print("Loading prompt examples from:", examplesPath)
        examples = pickle.load(fp)

    formatter = QuestionFormulator(examples, numExamplesPerPrompt, maxPromptSize=maxPromptSize)
    return formatter


