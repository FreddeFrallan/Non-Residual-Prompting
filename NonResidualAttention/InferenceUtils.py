from NonResidualAttention import InferenceContainers
from DecodingStrategies import WordInclusion
import tensorflow as tf


def HFSearchFunc(inferenceContainer, contextIDs, promptIDs, maxLen=64, numBeams=8):
    contextIDs = tf.convert_to_tensor(contextIDs, tf.int32)
    bSize, contextLen = tf.shape(contextIDs)
    inferenceContainer = InferenceContainers.HFDecodingMultiSentenceContainer(inferenceContainer, contextIDs, promptIDs,
                                                                              maxLen=maxLen,
                                                                              verbose=True, numBeams=numBeams)
    WordInclusion.CURRENT_INFERENCE_CONTAINER = inferenceContainer

    newMaxLen = int(contextLen.numpy() + maxLen)
    seqs = inferenceContainer.generate(contextIDs, max_length=newMaxLen, num_beams=numBeams,
                                       num_return_sequences=1,
                                       eos_token_id=None,
                                       pad_token_id=13, use_cache=False,
                                       repetition_penalty=1.25,
                                       )

    return [s.tolist() for s in seqs.numpy()]


def batchGeneration(inferenceContainer, tokenizer, contexts, promptIDs, maxSeqLen, numBeams):
    contextIDs = [tokenizer.encode(context) for context in contexts]
    seqs = HFSearchFunc(inferenceContainer, contextIDs, promptIDs, maxLen=maxSeqLen, numBeams=numBeams)
    generatedTexts = [tokenizer.decode(s).strip() for context, s in zip(contexts, seqs)]
    return generatedTexts
