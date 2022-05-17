
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

BEAM_STRATEGY = "beam"
NUCLEUS_STRATEGY = "nucleus"
GREEDY_STRATEGY = "greedy"
STRATEGIES = [BEAM_STRATEGY, NUCLEUS_STRATEGY, GREEDY_STRATEGY]

PAD_LEN = 400


class VanillaGPT2:

    def __init__(self, model_name="gpt2", generateStrategy="beam", promptTemplate="", batchGenerate=False):
        self.genLen = 50
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = TFAutoModelForCausalLM.from_pretrained(model_name,
                                                            eos_token_id=self.tokenizer.eos_token_id,
                                                            pad_token_id=self.tokenizer.pad_token_id)

        assert generateStrategy in STRATEGIES
        self.generateStrategy = generateStrategy
        self.promptTemplate = promptTemplate
        self.batchGenerate = batchGenerate

    def perplexityFromContextAndGenerated(self, contexts, generated):
        contexts = [c.strip("\n").strip() for c in contexts]
        generated = [g.strip("\n").strip() for g in generated]
        contextIds = [self.tokenizer.encode(c) for c in contexts]
        contextLens = [len(cIds) for cIds in contextIds]
        concatStrs = [" " if len(c) > 0 else "" for c in contexts]
        fullTexts = [c + cStr + g for c, cStr, g in zip(contexts, concatStrs, generated)]

        input_text = [self.tokenizer.encode(ft) for ft in fullTexts]

        numSamples = len(input_text)
        maxLen = max([len(inpText) for inpText in input_text])
        input_ids = np.ones((numSamples, maxLen), dtype=int) * self.tokenizer.pad_token_id
        attention_mask = np.zeros((numSamples, maxLen), dtype=int)
        # lens = np.zeros(numSamples)
        for i in range(numSamples):
            currInput = input_text[i]
            input_ids[i, :len(currInput)] = currInput
            attention_mask[i, :len(currInput)] = 1

        attention_mask = attention_mask[:, :-1]

        numSamples = len(input_ids)
        labels = np.copy(input_ids)[:, 1:]

        input_ids = input_ids[:, :-1]

        for i in range(len(labels)):
            labels[i] = [-100 if (x == self.tokenizer.pad_token_id) or (idx < (contextLens[i] - 1)) else x
                         for idx, x in enumerate(labels[i])]

        input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        outputs = self.model(input_ids, attention_mask=attention_mask)

        assert len(labels) == len(outputs.logits)
        perplexities = []
        for i in range(numSamples):
            y = tf.expand_dims(labels[i], axis=0)
            logits = tf.expand_dims(outputs.logits[i], axis=0)
            loss = self.model.compute_loss(labels=y, logits=logits)

            mean = tf.math.reduce_mean(loss)

            perplexity = tf.math.exp(mean)
            perplexities.append(perplexity.numpy())

        return perplexities
