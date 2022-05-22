from NonResidualAttention import NonResidualGPT2
from PostTransformation import PostTransformationModel, PostTransformConfig
import tensorflow as tf


class PromptModelSetup(tf.keras.Model):

    def __init__(self, modelBase, CLMWeightsPath=None, promptModelWeightsPath=None, postWeightsPath=None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.createModels(modelBase, CLMWeightsPath, promptModelWeightsPath)
        self.setupPostTransformation(postWeightsPath)

    def createModels(self, modelBase, CLMWeightsPath, promptModelWeightsPath):
        self.CLM = NonResidualGPT2.NonResidualGPT(modelBase, CLMWeightsPath)
        self.promptModel = NonResidualGPT2.NonResidualGPT(modelBase, promptModelWeightsPath)
        self.config = self.CLM.config

    def setupPostTransformation(self, postWeightsPath):
        if (postWeightsPath == None):
            postConfig = PostTransformConfig.PostTransformConfig(self.config.num_hidden_layers, self.config.n_head)
            self.postTransformation = PostTransformationModel.PositionalInvariantTransformation(postConfig)
        else:
            print("Loading Post weights:", postWeightsPath)
            self.postTransformation = PostTransformationModel.PositionalInvariantTransformation.from_pretrained(
                postWeightsPath)

    def get_output_embeddings(self):
        return self.CLM.model.get_output_embeddings()

    def generatePromptPast(self, promptIDs):
        hidden_states, inputs, output_shape, input_shape = self.promptModel.doubleDataPrepCall(self.CLM.oldCall,
                                                                                               self.CLM.customCallDataPrep,
                                                                                               input_ids=promptIDs,
                                                                                               use_cache=True,
                                                                                               output_attentions=True)

        output = self.promptModel.generateNormalPastIntermediate(hidden_states, inputs, output_shape, input_shape)
        past = output['past_key_values']

        return past

    def customCall(self, inputs):
        ids, promptIDs, att = inputs
        past = self.generatePromptPast(promptIDs)
        past = self.postTransformation(past)
        return self.CLM.call(ids, past, att).logits

    def call(self, inputs, training=None, mask=None):
        return self.customCall(inputs)
