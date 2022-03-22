from NonResidualAttention import CustomSearchFunction, PromptModelSetup
from transformers.models.gpt2.modeling_tf_gpt2 import shape_list
import transformers.generation_tf_utils as GenUtils
import tensorflow as tf
import numpy as np
import tqdm



class HFDecodingContainer(GenUtils.TFGenerationMixin):

    def __init__(self, promptModelSetup, promptIDs, numBeams=1, maxLen=64, verbose=True):
        self.numBeams = numBeams
        self.model = promptModelSetup
        self.numPrompts = len(promptIDs)
        self.fullBatchSize = self.numBeams * self.numPrompts
        self.promptBlockMask = np.ones((self.fullBatchSize, 1))
        # self.promptBlockMask = np.zeros((self.fullBatchSize, 1))

        self.past, self.promptMask, self.maxPromptLen = self.calculatePromptPast(promptIDs, self.numPrompts)
        self.config = self.model.config
        self.config.use_cache = True

        self.numPredCounter = 0
        self.seqPastLen = 0
        self.seqPast = None
        self.posIDs = None

        self.verbose = verbose
        if (verbose):
            self.progressBar = tqdm.tqdm(total=maxLen)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def calculatePromptPast(self, promptIDs, numPrompts):
        promptLens = [len(p) for p in promptIDs]
        maxPromptLen = np.max(promptLens)

        promptMask = np.zeros((numPrompts, maxPromptLen))
        for i, p in enumerate(promptIDs):
            promptMask[i][:len(p)] = 1

        paddedPromptIDs = [p + [0] * (maxPromptLen - len(p)) for p in promptIDs]
        paddedPromptIDs = tf.convert_to_tensor(paddedPromptIDs, tf.int32)

        return self.model.generatePromptPast(paddedPromptIDs), promptMask, maxPromptLen

    def getPast(self, batchSize):  # Gready with batching
        if (self.seqPast is None):
            return tf.repeat(self.past, self.numBeams, axis=2)
            # return self.past

    def generateAttMask(self, batchSize, seqSize):
        promptMask = np.repeat(self.promptMask, self.numBeams, axis=0) * self.promptBlockMask
        mask = tf.expand_dims(tf.expand_dims(promptMask, axis=1), axis=1)  # (Batch, 1, 1, PromptLen)
        seqMask = tf.repeat(mask, seqSize, axis=2)  # Repeat of the current Sequence
        return seqMask
        # return tf.repeat(seqMask, self.numBeams, axis=0)  # Repeat for all beams

    def getPosIDs(self, seqLen):
        return self.posIDs

    def __call__(self, *args, **kwargs):
        seqIDs = kwargs['input_ids']
        bSize, seqLen = shape_list(seqIDs)

        past = self.getPast(bSize)
        att = self.generateAttMask(bSize, seqLen)
        output = self.model.CLM.call(seqIDs, past, att, posIDs=self.getPosIDs(seqLen))

        if (self.verbose):
            self.progressBar.update(1)

        return output


class HFDecodingMultiSentenceContainer(HFDecodingContainer):

    def __init__(self, promptModelSetup, contextIDs, promptIDs, numBeams=1, maxLen=64, verbose=True):
        self.promptModelSetup = promptModelSetup
        self.CLM = self.promptModelSetup.CLM
        super().__init__(self.promptModelSetup, promptIDs, numBeams, maxLen, verbose)

        self.contextLen = int(tf.shape(contextIDs)[-1])
        # self.contextLen = 1

    def incorperateContextPast(self, contextIDs):
        contextIDs = contextIDs[:, :-1]  # Leave the final token for inference
        contextPast = self.CLM.generateNormalPastWithInputIDs(contextIDs)
        contextPastAtt = tf.ones(contextIDs.shape, tf.int32)

        self.preContextLen = tf.shape(contextIDs)[-1]
        self.past = tf.concat((self.past, contextPast), axis=-2)
        self.promptMask = tf.concat((self.promptMask, contextPastAtt), axis=-1)

    def calculatePromptPast(self, promptIDs, numPrompts, numBeams=1):
        promptLens = [len(p) for p in promptIDs]
        maxPromptLen = np.max(promptLens)

        promptMask = np.zeros((numPrompts, maxPromptLen))
        for i, p in enumerate(promptIDs):
            promptMask[i][:len(p)] = 1

        paddedPromptIDs = [p + [0] * (maxPromptLen - len(p)) for p in promptIDs]
        paddedPromptIDs = tf.convert_to_tensor(paddedPromptIDs, tf.int32)

        past = self.promptModelSetup.generatePromptPast(paddedPromptIDs)
        past = self.promptModelSetup.postTransformation(past)
        return past, promptMask, maxPromptLen

    def get_output_embeddings(self):
        return self.promptModelSetup.get_output_embeddings()

    def _generate_no_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, temperature, top_k, top_p,
                                 repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id,
                                 eos_token_id, batch_size, vocab_size, encoder_outputs, attention_mask, use_cache,
                                 return_dict_in_generate, **kwargs
                                 ) -> GenUtils.Union[GenUtils.TFGreedySearchOutput, GenUtils.TFSampleOutput, tf.Tensor]:
        return CustomSearchFunction._generate_no_beam_search(self, input_ids, cur_len, max_length, min_length,
                                                             do_sample,
                                                             temperature, top_k, top_p,
                                                             repetition_penalty, no_repeat_ngram_size, bad_words_ids,
                                                             pad_token_id,
                                                             eos_token_id, batch_size, vocab_size, encoder_outputs,
                                                             attention_mask, use_cache,
                                                             return_dict_in_generate,
                                                             self._create_next_token_logits_penalties, **kwargs)

    def _generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, early_stopping,
                              temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size,
                              bad_words_ids, pad_token_id, eos_token_id, batch_size, num_return_sequences,
                              length_penalty, num_beams, vocab_size, encoder_outputs, attention_mask,
                              use_cache, forced_bos_token_id, forced_eos_token_id, return_dict_in_generate,
                              **kwargs,
                              ) -> GenUtils.Union[GenUtils.TFGreedySearchOutput, GenUtils.TFSampleOutput, tf.Tensor]:
        return CustomSearchFunction._generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample,
                                                          early_stopping,
                                                          temperature, top_k, top_p, repetition_penalty,
                                                          no_repeat_ngram_size,
                                                          bad_words_ids, pad_token_id, eos_token_id, batch_size,
                                                          num_return_sequences,
                                                          length_penalty, num_beams, vocab_size, encoder_outputs,
                                                          attention_mask,
                                                          use_cache, forced_bos_token_id, forced_eos_token_id,
                                                          return_dict_in_generate, **kwargs)

    def _create_next_token_logits_penalties(self, input_ids, logits, repetition_penalty):
        return GenUtils._create_next_token_logits_penalties(input_ids[:, self.contextLen:], logits, repetition_penalty)
