import torch

from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)

from packaging import version

if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False


class GPT2NonResidualAttention(torch.nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer(
            "self_bias",
            torch.ones((max_positions, max_positions), dtype=torch.uint8).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def getCausualAttentionMask(self, query_length, key_length, nonResidual=False):
        if (nonResidual):
            nonResidualCasual = self.bias[:, :, key_length - query_length: key_length, 1:key_length + 1].bool()
            selfMask = self.self_bias[:, :, key_length - query_length: key_length, :1].bool()
            return torch.cat((nonResidualCasual, selfMask), dim=-1)
        else:
            return self.bias[:, :, key_length - query_length: key_length, :key_length].bool()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, promptModelMask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            if (promptModelMask is None):
                attMask = self.getCausualAttentionMask(query_length, key_length)
            else:
                attMask = promptModelMask
            attn_weights = torch.where(attMask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def nonResidualAttention(self, query, key, value, promptKeyPast, promptValuePast, textualKeyPast, textualValuePast,
                             promptMask=None, head_mask=None):
        promptWeights = torch.matmul(query, promptKeyPast.transpose(-1, -2))
        textWeights = torch.matmul(query, textualKeyPast.transpose(-1, -2))
        selfWeights = torch.sum(query * key, keepdim=True, dim=-1)

        attn_weights = torch.cat((promptWeights, textWeights, selfWeights), dim=-1)

        if self.scale_attn_weights:
            attn_weights = attn_weights / (textualValuePast.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            batchSize, query_length, key_length = query.size(0), query.size(-2), textualKeyPast.size(-2)
            causal_mask = self.getCausualAttentionMask(query_length, key_length, nonResidual=True)
            sampleCausalMask = causal_mask.repeat_interleave(batchSize, dim=0)
            # print(sampleCausalMask.shape, promptMask.shape)
            attentionMask = torch.cat((promptMask, sampleCausalMask), dim=-1)
            attn_weights = torch.where(attentionMask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(textualValuePast.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        wSelf, wRest = attn_weights[:, :, :, -1:], attn_weights[:, :, :, :-1]

        valuePast = torch.cat((promptValuePast, textualValuePast), dim=-2)
        outRest = torch.matmul(wRest, valuePast)
        outSelf = wSelf * value
        attn_output = outRest + outSelf

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False, output_attentions=False, nonResidual=False,
                promptPast=None, textualPast=None, promptMask=None):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if (nonResidual == False):
            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            if self.reorder_and_upcast_attn:
                attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask,
                                                                            head_mask)
            else:
                attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask,
                                                       promptModelMask=promptMask)
        else:
            present = None
            promptKey, promptValue = promptPast
            textualKey, textualValue = textualPast
            attn_output, attn_weights = self.nonResidualAttention(query, key, value, promptKey, promptValue, textualKey,
                                                                  textualValue, promptMask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

