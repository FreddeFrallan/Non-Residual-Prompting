from NonResidualAttentionPT import NonResidualGPT2
import torch
from torch import nn


class PositionalInvariantTransformation(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        # super().__init__(name='Post-Past-Constant', *args, **kwargs)
        postShape = (config.num_hidden_layers, 2, 1, config.n_head, 1, 64)
        self.postPastWeight = self.add_weight('Post-Past', postShape, torch.float32,
                                              tf.keras.initializers.Zeros(),
                                              trainable=True)

    def call(self, past, training=None, mask=None):
        return past + self.postPastWeight


class PromptModelSetup(nn.Module):

    def __init__(self, modelBase, CLMWeightsPath=None, promptModelWeightsPath=None, postWeightsPath=None, *args,
                 **kwargs):
        super().__init__()
        # super().__init__(*args, **kwargs)

        self.CLM = NonResidualGPT2.NonResidualGPTPT(modelBase, CLMWeightsPath)
        self.promptModel = NonResidualGPT2.NonResidualGPTPT(modelBase, promptModelWeightsPath)
        self.config = self.CLM.config

        self.postTransformation = PositionalInvariantTransformation(self.config)
        if (postWeightsPath is not None):
            print("Loading Post weights:", postWeightsPath)
            self.postTransformation.load_weights(postWeightsPath).expect_partial()

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
