import argparse
import Utils
from DecodingStrategies import WordInclusion
from NonResidualAttention import InferenceUtils


def generate(args):
    model, tokenizer, maxSeqLen, promptGenerator = Utils.loadPaperGPT2LargeModel()

    sentLen, seqLen, numBeams, context = args.sentence_length, args.generate_length, args.num_beams, args.context

    targetWords = [args.target_words]

    promptIDs = [promptGenerator.createPromptIDs(words, sentLen) for words in targetWords]
    contextLen = len(tokenizer.encode(context))
    WordInclusion.setCurrentInclusionWords(targetWords, sentLen, contextLen=contextLen, numBeams=numBeams)

    sampleContexts = [context for _ in targetWords]
    result = InferenceUtils.batchGeneration(model, tokenizer, sampleContexts, promptIDs, seqLen, numBeams)
    for txt, words in zip(result, targetWords):
        print("------------", words)
        print(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_words', nargs='+', default=[],
                        help="The target words that should be included in the generated text.")
    parser.add_argument("--context", type=str, default="The",
                        help="The context that the model continues to generate from.")
    parser.add_argument("--sentence_length", type=int, default=15,
                        help="The sentence length that the model will be instructed with.")
    parser.add_argument("--generate_length", type=int, default=32,
                        help="The number of tokens in the resulting text. Maximum of 128.")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="The number of beams to use within beam search.")

    args = parser.parse_args()

    assert len(args.target_words) > 0
    assert args.sentence_length > 0
    assert args.generate_length > 0
    assert args.num_beams > 0
    args.generate_length = min(args.generate_length, 128)

    generate(args)
