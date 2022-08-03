import SimpleNonResidualGPT2
import transformers
import torch


class PromptTuningGPT2Setup(torch.nn.Module):

    def __init__(self, modelBase):
        super().__init__()
        self.config = transformers.GPT2Config.from_pretrained(modelBase)
        self.clm = SimpleNonResidualGPT2.NonResidualGPT2LMHeadModel.from_pretrained(modelBase)
        self.promptModel = transformers.GPT2Model.from_pretrained(modelBase)

    def prepareTrainingParameters(self):
        for p in self.clm.parameters():
            p.requires_grad = False
        for p in self.promptModel.parameters():
            p.requires_grad = True

    def generatePromptAndTextualPast(self, inputIDs, promptIDs):
        promptPast = self.promptModel.forward(promptIDs, use_cache=True)['past_key_values']
        textualPast = self.clm.transformer.forward(inputIDs, use_cache=True)['past_key_values']
        return promptPast, textualPast

    def forward(self, inputIDs, promptIDs, promptMask, maskedLabels=None):
        promptPast, textualPast = self.generatePromptAndTextualPast(inputIDs, promptIDs)
        return self.clm.forward(inputIDs, promptPast=promptPast, textualPast=textualPast,
                                  promptMask=promptMask, labels=maskedLabels)
