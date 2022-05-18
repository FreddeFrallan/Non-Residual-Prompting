# from transformers.models.gpt2.modeling_tf_gpt2 import shape_list, TFBaseModelOutputWithPast, input_processing, \
#     TFCausalLMOutputWithPast

# TFGPT2Model: https://github.com/huggingface/transformers/blob/d6b8e9cec7301ba02f642588a6f12e78ec3b9798/src/transformers/models/gpt2/modeling_tf_gpt2.py#L707
from typing import Optional, Tuple, Union

from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from NonResidualAttentionPT import NonResidualAttention
# import tensorflow as tf
import transformers
# import numpy as np
import torch
from torch import nn


class NonResidualGPTPT(nn.Module):
    '''
    A slightly modified version of GPT-2 to allow Non-residual prompt instructions.
    Although this file contains a lot of code the majority of it is overriding functions in GPT-2 with minor changes.

    For example, the non-residual attention mask is passed as the attention_mask variable,
    meaning that the data prep functions need to be overridden to allow for non-uniform attention masks.
    '''

    # def __init__(self, modelBase, preTrainedPath=None, initModelParameters=True, *args, **kwargs):
    def __init__(self, modelBase, preTrainedPath=None, initModelParameters=True, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.createModel(modelBase, preTrainedPath)

        if (initModelParameters):
            self.initParamaters()

    def createModel(self, modelBase, preTrainedPath):
        if (preTrainedPath is not None):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(preTrainedPath)
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(modelBase)
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
    """
    def customCallDataPrep(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                           head_mask=None, inputs_embeds=None, use_cache=None, output_attentions=None,
                           output_hidden_states=None, return_dict=None, training=False, **kwargs):

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
            inputs["input_ids"] = torch.reshape(inputs["input_ids"], [-1, input_shape[-1]])
        elif inputs["inputs_embeds"] is not None:
            input_shape = shape_list(inputs["inputs_embeds"])[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs["past"] is None:
            inputs["past"] = [None] * self.totalNumLayers

        # ***** Custom Hack *****
        # ***** This had to be hacked, as we want to input a non-uniform mask.
        if inputs["attention_mask"] is not None:
            # inputs["attention_mask"] = tf.cast(inputs["attention_mask"], tf.float32)
            inputs["attention_mask"] = inputs["attention_mask"].type(torch.float32)

        # ***** Custom Hack *****
        # ***** This had to be hacked, allowing us to NOT modify the positions of the IDs, although we have a past
        if inputs["position_ids"] is None:
            inputs["position_ids"] = tf.expand_dims(torch.arange(0, input_shape[-1]), axis=0)

        if inputs["head_mask"] is not None:
            raise NotImplementedError
        else:
            inputs["head_mask"] = [None] * self.totalNumLayers

        inputs["position_ids"] = torch.reshape(inputs["position_ids"], [-1, shape_list(inputs["position_ids"])[-1]])

        if inputs["inputs_embeds"] is None:
            inputs["inputs_embeds"] = self.wte(inputs["input_ids"], mode="embedding")

        position_embeds = tf.gather(self.wpe, inputs["position_ids"])

        if inputs["token_type_ids"] is not None:
            inputs["token_type_ids"] = torch.reshape(
                inputs["token_type_ids"], [-1, shape_list(inputs["token_type_ids"])[-1]]
            )
            token_type_embeds = self.wte(inputs["token_type_ids"], mode="embedding")
        else:
            # TODO: is this equivalent to tf.constant()?
            token_type_embeds = torch.tensor(0.0)

        # position_embeds = tf.cast(position_embeds, dtype=inputs["inputs_embeds"].dtype)
        position_embeds = position_embeds.type(inputs["inputs_embeds"].dtype)
        # token_type_embeds = tf.cast(token_type_embeds, dtype=inputs["inputs_embeds"].dtype)
        token_type_embeds = token_type_embeds.type(inputs["inputs_embeds"].dtype)
        hidden_states = inputs["inputs_embeds"] + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=inputs["training"])

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        return hidden_states, inputs, input_shape, output_shape
        # return hidden_states, inputs, output_shape, input_shape
    """

    def customForward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            fredrik_cache=False
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        # ***** Custom Hack *****
        # ***** This had to be hacked, allowing us to NOT modify the positions of the IDs, although we have a past
        if position_ids is None:
            # position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            # position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            """
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    # logger.warning(
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
            """
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True or True:  # Always store cache
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    """
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
    """


    # *********************************
    # ***** Custom Call Functions *****
    # *********************************
    """
    def customCall(self, hidden_states, inputs, input_shape, output_shape):
        presents = ()  # if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None

        # ***** Custom Hack *****
        # Removed the zip-iteration of the past, allowing us to pass it in the form of a tensor, enabling TF.Function
        for i, block in enumerate(self.h):
            layer_past = inputs["past"][i]

            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (torch.reshape(hidden_states, output_shape),)

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

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents,
                                         hidden_states=all_hidden_states, attentions=all_attentions)
    """

    """
    def customCallNoCache(self, hidden_states, inputs, input_shape, output_shape):
        presents = None  # presents = () if inputs["use_cache"] else None
        all_attentions = () if inputs["output_attentions"] else None
        all_hidden_states = () if inputs["output_hidden_states"] else None

        # ***** Custom Hack *****
        # Removed the zip-iteration of the past, allowing us to pass it in the form of a tensor
        for i, block in enumerate(self.h):
            layer_past = inputs["past"][i]

            if inputs["output_hidden_states"]:
                all_hidden_states = all_hidden_states + (torch.reshape(hidden_states, output_shape),)

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
            all_attentions = tuple(torch.reshape(t, attention_output_shape) for t in all_attentions)

        if not inputs["return_dict"]:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents,
                                         hidden_states=all_hidden_states, attentions=all_attentions)
    """

    def customCallPost(self, output, output_shape):  # Can be used for No-Cache call as well
        hidden_states = output['last_hidden_state']
        hidden_states = self.model.transformer.ln_f(hidden_states)

        # ********** From the outer call func
        logits = self.model.transformer.wte(hidden_states, mode="linear")
        return CausalLMOutputWithCrossAttentions(loss=None, logits=logits, past_key_values=output.past_key_values,
                                        hidden_states=output.hidden_states, attentions=output.attentions)

    # ***************************************
    # ***** Output Generation Functions *****
    # ***************************************
    # def generateNormalPastIntermediate(self, hidden_states, inputs, output_shape, input_shape):
    #     return self.customCall(hidden_states, inputs, output_shape, input_shape)

    def generateNormalPastWithInputIDs(self, inputIDs):
        return self.model.call(input_ids=inputIDs, use_cache=True, output_attentions=True)['past_key_values']
        # return self.model.call({'input_ids': inputIDs, 'use_cache': True, 'output_attentions': True})['past_key_values']

    # def generatePredictionGivenPast(self, inputIDs, promptPast, promptAtt, normalPast):
    #     customPast = self.createCombinedPast(normalPast, promptPast)
    #     return self.model.callNoCache(
    #         {'input_ids': inputIDs, 'use_cache': False, 'past': customPast, 'attention_mask': promptAtt,
    #          'output_attentions': True})

    def generateOutputGivenNormalTextPast(self, inputIDs, promptPast, promptAtt, normalPast, posIDs=None):
        customPast = self.createCombinedPast(normalPast, promptPast)
        inData = {'input_ids': inputIDs, 'use_cache': False, 'past': customPast, 'attention_mask': promptAtt,
                  'output_attentions': True, 'position_ids': posIDs}
        # hidden_states, inputs, output_shape, input_shape = self.customCallNoCacheInnerPrepData(inData)
        # Fredrik's Legacy: order of input_shape & output_shape mustn't be unswitched
        hidden_states, inputs, output_shape, input_shape = self.customCallDataPrep(inData)

        return self.customCallNoCache(hidden_states, inputs, output_shape,
                                      input_shape), inputs, output_shape, input_shape

    def generateNormalTextPast(self, inputIDs, posIDs=None):
        # Start by initializing and preparing the data
        # hidden_states, inputs, output_shape, input_shape = self.doubleDataPrepCall(self.oldCall,
        # hidden_states, inputs, input_shape, output_shape = self.doubleDataPrepCall(self.oldCall,
        #                                                                            self.customCallDataPrep,
        #                                                                            input_ids=inputIDs, use_cache=True,
        #                                                                            output_attentions=True,
        #                                                                            position_ids=posIDs)
        return self.customForward(inputIDs)
        # return self.generateNormalPastIntermediate(hidden_states, inputs, output_shape,
        #                                            input_shape), inputs, output_shape, input_shape

    # def call(self, inputIDs, promptPast, promptAtt, posIDs=None, training=None, mask=None):
    def forward(self, inputIDs, promptPast, promptAtt, posIDs=None, training=None, mask=None):
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

    """
    def createRandomPast(self, batchSize, seqLen, pastSize=5, nHeads=12, nLayers=12, embSize=64):
        randomPast = []
        for i in range(nLayers):
            rv = torch.normal(0.0, 1.0, (batchSize, nHeads, pastSize, embSize))
            rk = torch.normal(0.0, 1.0, (batchSize, nHeads, pastSize, embSize))
            randomPast.append((rv, rk))

        randomAttn = torch.zeros((batchSize, 1, seqLen, pastSize))
        # TODO: is stack signature equivalent for PyTorch?
        return torch.stack(randomPast), randomAttn
    """

    def createCombinedPast(self, past, promptPast):
        past = torch.stack(past)
        return torch.cat((promptPast, past), -2)

