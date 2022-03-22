from transformers.models.gpt2.modeling_tf_gpt2 import shape_list, TFBaseModelOutputWithPast, input_processing, \
    TFCausalLMOutputWithPast
from NonResidualAttention import NonResidualAttention
import tensorflow as tf
import transformers
import numpy as np


class NonResidualGPT(tf.keras.Model):
    '''
    A slightly modified version of GPT-2 to allow Non-residual prompt instructions.
    Although this file contains a lot of code the majority of it is overriding functions in GPT-2 with minor changes.

    For example, the non-residual attention mask is passed as the attention_mask variable,
    meaning that the data prep functions need to be overridden to allow for non-uniform attention masks.
    '''

    def __init__(self, modelBase, preTrainedPath=None, initModelParameters=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.createModel(modelBase, preTrainedPath)

        if (initModelParameters):
            self.initParamaters()

    def createModel(self, modelData, preTrainedPath):
        if (preTrainedPath is not None):
            self.model = transformers.TFAutoModelForCausalLM.from_pretrained(preTrainedPath)
        else:
            self.model = transformers.TFAutoModelForCausalLM.from_pretrained(modelData)
        self.totalNumLayers = self.model.config.n_layer
        self.model.config.output_attentions = True

    def initParamaters(self):
        self.initNonResidualAttentionLayers()
        self.initCustomCalls()

    def initNonResidualAttentionLayers(self):
        for block in self.model.transformer.h:
            NonResidualAttention.convertCausualAttentionLayerIntoNonResidualAttention(block.attn)

    def initCustomCalls(self):
        self.config = self.model.config
        self.h = self.model.transformer.h
        self.wte = self.model.transformer.wte
        self.wpe = self.model.transformer.wpe
        self.drop = self.model.transformer.drop
        self.ln_f = self.model.transformer.ln_f
        self.oldCall = self.model.transformer.call
        self.num_hidden_layers = self.model.transformer.num_hidden_layers

    # **************************************
    # ***** Custom Data Prep Functions *****
    # **************************************
    def customCallDataPrep(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                           head_mask=None, inputs_embeds=None, use_cache=None, output_attentions=None,
                           output_hidden_states=None, return_dict=None, training=False, **kwargs):

        inputs = input_processing(func=self.oldCall, config=self.config, input_ids=input_ids, past=past,
                                  attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states, return_dict=return_dict, training=training,
                                  kwargs_call=kwargs,
                                  )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
            inputs["input_ids"] = tf.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            inputs["past"] = [None] * self.totalNumLayers

        # ***** Custom Hack *****
        # ***** This had to be hacked, as we want to input a non-uniform mask.
        if inputs["attention_mask"] is not None:
            inputs["attention_mask"] = tf.cast(inputs["attention_mask"], tf.float32)

        # ***** Custom Hack *****
        # ***** This had to be hacked, allowing us to NOT modify the positions of the IDs, although we have a past
        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(tf.range(0, input_shape[-1]), axis=0)

        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.totalNumLayers

        inputs["position_ids"] = tf.reshape(inputs["position_ids"], [-1, shape_list(inputs["position_ids"])[-1]])

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        position_embeds = tf.gather(self.wpe, inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = tf.reshape(
                inputs["token_type_ids"], [-1, shape_list(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs["inputs_embeds"].dtype)
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        return hidden_states, inputs, output_shape, input_shape

    def customCallNoCacheInnerPrepData(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None,
                                       position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None,
                                       output_attentions=None, output_hidden_states=None, return_dict=None,
                                       training=False, **kwargs):

        inputs = input_processing(func=self.oldCall, config=self.config, input_ids=input_ids, past=past,
                                  attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                  use_cache=use_cache, output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states, return_dict=return_dict, training=training,
                                  kwargs_call=kwargs,
                                  )

        if inputs["input_ids"] is not None and inputs["inputs_embeds"] is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs["input_ids"] is not None:
            input_shape = shape_list(inputs["input_ids"])
            inputs["input_ids"] = tf.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            inputs["past"] = [None] * self.totalNumLayers

        # ***** Custom Hack *****
        # ***** This had to be hacked, as we want to input a non-uniform mask.
        if inputs["attention_mask"] is not None:
            inputs["attention_mask"] = tf.cast(inputs["attention_mask"], tf.float32)

        # ***** Custom Hack *****
        # ***** This had to be hacked, allowing us to NOT modify the positions of the IDs, although we have a past
        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(tf.range(0, input_shape[-1]), axis=0)

        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.totalNumLayers

        inputs["position_ids"] = tf.reshape(inputs["position_ids"], [-1, shape_list(inputs["position_ids"])[-1]])

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        position_embeds = tf.gather(self.wpe, inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = tf.reshape(
                inputs["token_type_ids"], [-1, shape_list(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs["inputs_embeds"].dtype)
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        return hidden_states, inputs, input_shape, output_shape

    def doubleDataPrepCall(self, outerFuncID, innerPrepFunc, input_ids=None, past=None,
                           attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                           inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                           return_dict=None, training=False, **kwargs):

        inputs = input_processing(
            func=outerFuncID, config=self.config, input_ids=input_ids, past=past,
            attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
            training=training, kwargs_call=kwargs, )

        return innerPrepFunc(input_ids=inputs["input_ids"], past=inputs["past"],
                             attention_mask=inputs["attention_mask"], token_type_ids=inputs["token_type_ids"],
                             position_ids=inputs["position_ids"], head_mask=inputs["head_mask"],
                             inputs_embeds=inputs["inputs_embeds"], use_cache=inputs["use_cache"],
                             output_attentions=inputs["output_attentions"],
                             output_hidden_states=inputs["output_hidden_states"], return_dict=inputs["return_dict"],
                             training=inputs["training"])

    # *********************************
    # ***** Custom Call Functions *****
    # *********************************
    def customCall(self, hidden_states, inputs, input_shape, output_shape):
        presents = ()  # if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None

        # ***** Custom Hack *****
        # Removed the zip-iteration of the past, allowing us to pass it in the form of a tensor, enabling TF.Function
        for i, block in enumerate(self.h):
            layer_past = inputs["past"][i]

            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(hidden_states, layer_past, inputs["attention_mask"], inputs["head_mask"][i], True,
                            inputs["output_attentions"], training=inputs["training"])

            hidden_states, present = outputs[:2]
            presents = presents + (present,)  # Always store cache

            if inputs["output_attentions"]:
                all_attentions = all_attentions + (outputs[2],)

        if inputs["output_hidden_states"]:  # Add last hidden state
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents,
                                         hidden_states=all_hidden_states, attentions=all_attentions)

    def customCallNoCache(self, hidden_states, inputs, input_shape, output_shape):
        presents = None  # presents = () if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None

        # ***** Custom Hack *****
        # Removed the zip-iteration of the past, allowing us to pass it in the form of a tensor
        for i, block in enumerate(self.h):
            layer_past = inputs["past"][i]

            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block(hidden_states, layer_past, inputs["attention_mask"], inputs["head_mask"][i], False,
                            inputs["output_attentions"], training=inputs["training"])

            hidden_states, present = outputs[:2]
            if inputs["output_attentions"]:
                all_attentions = all_attentions + (outputs[2],)

        # Add last hidden state
        if inputs["output_hidden_states"]:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if inputs["output_attentions"]:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents,
                                         hidden_states=all_hidden_states, attentions=all_attentions)

    def customCallPost(self, output, output_shape):  # Can be used for No-Cache call as well
        hidden_states = output['last_hidden_state']
        hidden_states = self.model.transformer.ln_f(hidden_states)

        # ********** From the outer call func
        logits = self.model.transformer.wte(hidden_states, mode="linear")
        return TFCausalLMOutputWithPast(loss=None, logits=logits, past_key_values=output.past_key_values,
                                        hidden_states=output.hidden_states, attentions=output.attentions)

    # ***************************************
    # ***** Output Generation Functions *****
    # ***************************************
    def generateNormalPastIntermediate(self, hidden_states, inputs, output_shape, input_shape):
        return self.customCall(hidden_states, inputs, output_shape, input_shape)

    def generateNormalPastWithInputIDs(self, inputIDs):
        return self.model.call({'input_ids': inputIDs, 'use_cache': True, 'output_attentions': True})['past_key_values']

    def generatePredictionGivenPast(self, inputIDs, promptPast, promptAtt, normalPast):
        customPast = self.createCombinedPast(normalPast, promptPast)
        return self.model.callNoCache(
            {'input_ids': inputIDs, 'use_cache': False, 'past': customPast, 'attention_mask': promptAtt,
             'output_attentions': True})

    def generateOutputGivenNormalTextPast(self, inputIDs, promptPast, promptAtt, normalPast, posIDs=None):
        customPast = self.createCombinedPast(normalPast, promptPast)
        inData = {'input_ids': inputIDs, 'use_cache': False, 'past': customPast, 'attention_mask': promptAtt,
                  'output_attentions': True, 'position_ids': posIDs}
        hidden_states, inputs, output_shape, input_shape = self.customCallNoCacheInnerPrepData(inData)

        return self.customCallNoCache(hidden_states, inputs, output_shape,
                                      input_shape), inputs, output_shape, input_shape

    def generateNormalTextPast(self, inputIDs, posIDs=None):
        # Start by initializing and preparing the data
        hidden_states, inputs, output_shape, input_shape = self.doubleDataPrepCall(self.oldCall,
                                                                                   self.customCallDataPrep,
                                                                                   input_ids=inputIDs, use_cache=True,
                                                                                   output_attentions=True,
                                                                                   position_ids=posIDs)
        return self.generateNormalPastIntermediate(hidden_states, inputs, output_shape,
                                                   input_shape), inputs, output_shape, input_shape

    def call(self, inputIDs, promptPast, promptAtt, posIDs=None, training=None, mask=None):
        '''
        Given the textual InputIDs, and the prompt models key-values, this function performs two forward passes.
        The first pass is a completeley normal GPT-2 pass, to generate the textual stream
        The second pass generates the Non-residual prediction, which also attends the prompt key-values
        '''

        # The textual forward pass
        normalOutput, _, _, _ = self.generateNormalTextPast(inputIDs, posIDs)
        normalPast = normalOutput['past_key_values']

        # The Non-residual forward pass
        output, _, output_shape, _ = self.generateOutputGivenNormalTextPast(inputIDs, promptPast, promptAtt, normalPast,
                                                                            posIDs)
        output = self.customCallPost(output, output_shape)  # Generate Logits from the hidden states
        output['past_key_values'] = normalPast
        return output

    # ****************************
    # ***** Helper Functions *****
    # ****************************

    def createRandomPast(self, batchSize, seqLen, pastSize=5, nHeads=12, nLayers=12, embSize=64):
        randomPast = []
        for i in range(nLayers):
            rv = tf.random.normal((batchSize, nHeads, pastSize, embSize))
            rk = tf.random.normal((batchSize, nHeads, pastSize, embSize))
            randomPast.append((rv, rk))

        randomAttn = tf.zeros((batchSize, 1, seqLen, pastSize))
        return tf.stack(randomPast), randomAttn

    def createCombinedPast(self, past, promptPast):
        past = tf.stack(past)
        return tf.concat((promptPast, past), axis=-2)


def sanityCheck():
    base = 'gpt2'
    normalModel = transformers.TFAutoModelForCausalLM.from_pretrained(base)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    pModel = NonResidualGPT(base)

    texts = [
        'This is the first sentence.',
        'This is a slightly longer sentence, that follows the first, just longer.'
    ]
    inputsIDs = [tokenizer.encode(txt) for txt in texts]
    lens = [len(ids) for ids in inputsIDs]
    maxLen = np.max(lens)
    paddedIDs = [ids + [0] * (maxLen - len(ids)) for ids in inputsIDs]
    paddedIDs = tf.convert_to_tensor(paddedIDs, tf.int32)
    print("Input:", paddedIDs.shape)

    normalOutput = normalModel.call({'input_ids': paddedIDs})['logits']
    randPast, randAttn = pModel.createRandomPast(len(inputsIDs), maxLen)
    pastOutput = pModel.call(paddedIDs, randPast, randAttn)['logits']
    print(normalOutput.shape, pastOutput.shape)

    diff = tf.reduce_mean(tf.abs(normalOutput - pastOutput))
    print("Diff:", diff.numpy())


def main():
    sanityCheck()
