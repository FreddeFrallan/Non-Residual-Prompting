from transformers.models.gpt2.modeling_tf_gpt2 import shape_list
import tensorflow as tf


class NonResidualAttention:

    def __init__(self, attnLayer, **kwargs):
        super().__init__(**kwargs)
        self.originialLayer = attnLayer
        self._overrideFunctionalityInOriginalLayer(attnLayer)

    def _overrideFunctionalityInOriginalLayer(self, attnLayer):
        self.oldCall = attnLayer.call
        attnLayer.call = self.customCall

    @staticmethod
    def causal_past_attention_mask(nd, ns, dtype):
        '''
        Creates a normal, triangular, causal attention mask for attending to the textual stream.
        The only difference from a normal causal attention mask is that we don't self-attend, since this is reserved for
        the non-residual stream.
        '''
        i = tf.range(nd)[:, None]
        j = tf.range(ns)

        # This is changed to remove self-attention to the text stream
        # Original code: m = i >= j - ns + nd
        m = i > j - ns + nd

        m = tf.cast(m, dtype)
        m = tf.concat((m, tf.ones((nd, 1))), axis=-1)
        return m

    def _customAttn(self, keyPast, valuePast, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, keyPast, transpose_b=True)
        wSelf = tf.reduce_sum(q * k, axis=-1, keepdims=True)
        w = tf.concat((w, wSelf), axis=-1)

        if self.originialLayer.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        bSize, nHeads, nd, ns = shape_list(q)
        b = self.causal_past_attention_mask(nd, nd, dtype=w.dtype)
        b = tf.expand_dims(tf.expand_dims(b, axis=0), axis=0)

        if (attention_mask != None):
            b = tf.repeat(b, bSize, axis=0)
            b = tf.concat((attention_mask, b), axis=-1)  # Concat the Non-residual attention mask with the causal mask

        w = w * b - 1e4 * (1 - b)
        w = tf.nn.softmax(w, axis=-1)
        w = self.originialLayer.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        wSelf, wRest = w[:, :, :, -1:], w[:, :, :, :-1]
        finalValues = tf.matmul(wRest, valuePast) + wSelf * v

        outputs = [finalValues]
        if output_attentions:
            outputs.append(w)
        return outputs

    def customCall(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        '''
        The Non-Residual attention mask is passed via the "attention_mask" parameter. Which is later concatenated with
        the normal causal attention mask.
        '''
        if (use_cache):  # Hack to make it possible to use the layer without any non-residuals with TF.Function
            return self.oldCall(x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False)

        x = self.originialLayer.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.originialLayer.split_heads(query)
        key = self.originialLayer.split_heads(key)
        value = self.originialLayer.split_heads(value)
        past_key, past_value = tf.unstack(layer_past, axis=0)

        # to cope with keras serialization
        present = (None,)

        attn_outputs = self._customAttn(past_key, past_value, query, key, value, attention_mask, head_mask,
                                        output_attentions, training=training)
        a = attn_outputs[0]

        a = self.originialLayer.merge_heads(a)
        a = self.originialLayer.c_proj(a)
        a = self.originialLayer.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


def convertCausualAttentionLayerIntoNonResidualAttention(attnLayer):
    # This converts the original, normal causal attention layer, by overriding its call function
    # The old call functionality can still be achieved be setting use_cache=False
    NonResidualAttention(attnLayer)
