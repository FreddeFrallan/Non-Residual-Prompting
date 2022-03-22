from enum import Enum
import tqdm


class WordTypes(Enum):
    NEW_WORD = 0
    CONTINUATION = 1
    SPECIAL_SIGN = 2


class VocabularyManager:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.index2wordType = self._setupVocabLabels()

    def _setupVocabLabels(self, context='The'):
        vocabSize = len(self.tokenizer.get_vocab())
        contextIDs = self.tokenizer.encode(context)
        index2wordType = {}
        for i in tqdm.tqdm(range(vocabSize), 'Generating Vocab Mapping'):
            txt = self.tokenizer.decode(contextIDs + [i])
            startChar = txt[len(context)]
            if (startChar == ' '):
                index2wordType[i] = WordTypes.NEW_WORD
            elif (str.isalpha(startChar)):
                index2wordType[i] = WordTypes.CONTINUATION
            else:
                index2wordType[i] = WordTypes.SPECIAL_SIGN

        return index2wordType

    def getWordType(self, index):
        return self.index2wordType[index]


def main():
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    vocabManager = VocabularyManager(tokenizer)
