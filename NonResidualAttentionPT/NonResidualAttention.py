# Not present in PyTorch, does it handle dynamic shapes out of the box?
# from transformers.models.gpt2.modeling_gpt2 import shape_list
from typing import Optional, Tuple, Union

import torch
from torch import nn


class NonResidualAttention:

    def __init__(self, attnLayer, **kwargs):
        super().__init__(**kwargs)
        self.originialLayer = attnLayer
        self._overrideFunctionalityInOriginalLayer(attnLayer)

    def _overrideFunctionalityInOriginalLayer(self, attnLayer):
        # TODO: call is forward in PyTorch right?
        # self.oldCall = attnLayer.call
        # attnLayer.call = self.customCall
        self.oldForward = attnLayer.foward
        attnLayer.forward = self.customForward

    @staticmethod
    def causal_past_attention_mask(nd, ns, dtype):
        '''
        Creates a normal, triangular, causal attention mask for attending to the textual stream.
        The only difference from a normal causal attention mask is that we don't self-attend, since this is reserved for
        the non-residual stream.
        '''
        # TODO: Does this work instead of casting afterwards?
        i = torch.arange(0, nd)[:, None]
        j = torch.arange(0, ns)

        # This is changed to remove self-attention to the text stream
        # Original code: m = i >= j - ns + nd
        # m = i > j - ns + nd
        m = i > j - ns + nd

        m = m.type(dtype)

        ones = torch.ones((nd, 1))

        m = torch.cat((m, ones), -1)
        # m = tf.concat((m, tf.ones((nd, 1))), axis=-1)
        return m


    def _customAttn(self, keyPast, valuePast, query, key, value, attention_mask=None, head_mask=None,
                    output_attentions=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        wSelf = torch.sum(query * key, -1, keepdim=True)
        attn_weights = torch.cat((attn_weights, wSelf), -1)

        if self.originialLayer.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.originialLayer.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.originialLayer.layer_idx + 1)

        if not self.originialLayer.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            # causal_mask = self.originialLayer.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            causal_mask = self.causal_past_attention_mask(query_length, key_length, dtype=attn_weights.dtype)
            # causal_mask = causal_mask.expand()  # TODO?
            # causal_mask = torch.reshape(causal_mask, (1, 1, query_length, key_length))
            causal_mask = torch.reshape(causal_mask, (1, 1, query_length, -1))

        if attention_mask is not None:
            bSize = query.size(0)
            causal_mask = causal_mask.repeat(bSize, 1, 1, 1)
            # Concat the Non-residual attention mask with the causal mask
            causal_mask = torch.cat((attention_mask, causal_mask), -1)
            # Apply the attention mask
            attn_weights = torch.where(causal_mask, attn_weights,
                                       self.originialLayer.masked_bias.to(attn_weights.dtype))
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.originialLayer.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        wSelf, wRest = attn_weights[:, :, :, -1:], attn_weights[:, :, :, :-1]
        attn_output = torch.matmul(wRest, valuePast) + wSelf * value
        # attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # def customCall(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
    def customForward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        if use_cache:  # Hack to make it possible to use the layer without any non-residuals with TF.Function
            # return self.oldCall(hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False)
            return self.oldForward(hidden_states, layer_past, attention_mask, head_mask, encoder_hidden_states,
                                encoder_attention_mask, use_cache, output_attentions)

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.originialLayer.q_attn(hidden_states)
            key, value = self.originialLayer.c_attn(encoder_hidden_states).split(self.originialLayer.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.originialLayer.c_attn(hidden_states).split(self.originialLayer.split_size, dim=2)

        query = self.originialLayer._split_heads(query, self.originialLayer.num_heads, self.originialLayer.head_dim)
        key = self.originialLayer._split_heads(key, self.originialLayer.num_heads, self.originialLayer.head_dim)
        value = self.originialLayer._split_heads(value, self.originialLayer.num_heads, self.originialLayer.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.originialLayer.reorder_and_upcast_attn:
            attn_output, attn_weights = self.originialLayer._upcast_and_reordered_attn(query, key, value,
                                                                                       attention_mask, head_mask)
        else:
            # attn_output, attn_weights = self.originialLayer._attn(query, key, value, attention_mask, head_mask)
            attn_output, attn_weights = self._customAttn(past_key, past_value, query, key, value, attention_mask,
                                                         head_mask, output_attentions)

        attn_output = self.originialLayer._merge_heads(attn_output, self.originialLayer.num_heads,
                                                       self.originialLayer.head_dim)
        attn_output = self.originialLayer.c_proj(attn_output)
        attn_output = self.originialLayer.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def convertCausualAttentionLayerIntoNonResidualAttention(attnLayer):
    # This converts the original, normal causal attention layer, by overriding its call function
    # The old call functionality can still be achieved be setting use_cache=False
    NonResidualAttention(attnLayer)


if __name__ == '__main__':
    causal_mask = NonResidualAttention.causal_past_attention_mask(4, 4, torch.float32)
    causal_mask = torch.reshape(causal_mask, (1, 1, 4, -1))
    causal_mask = causal_mask.repeat(2, 1, 1, 1)
    print(causal_mask)
