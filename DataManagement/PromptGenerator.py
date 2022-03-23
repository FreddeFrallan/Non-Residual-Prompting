
class PromptGenerator:

    def __init__(self, tokenizer, maxPromptLen, promptTemplatePath="DataManagement/PromptTemplate.txt"):
        self.tokenizer = tokenizer
        self.maxPromptLen = maxPromptLen
        with open(promptTemplatePath, 'r') as fp:
            self.promptTemplate = fp.read()

    def createPromptIDs(self, targetWords, sentLen):
        targetWordsList = str.join(', ', targetWords)
        txt = self.promptTemplate + "{} words and target words {}.\n".format(sentLen, targetWordsList)
        ids = self.tokenizer.encode(txt)
        if (len(ids) > self.maxPromptLen):
            print("****WARNING*** Your prompt instruction is longer than the max length {}.".format(self.maxPromptLen))
        return ids

