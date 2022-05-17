import argparse
import datetime
import json
import time
import os

from BenchMarkSuite import benchMetrics
from EvaluationDatasetLoaderHF import readEvaluationDatasetByName, EVALUATION_DATASET_NAMES

GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def readGeneratedTexts(filePath):
    with open(filePath, 'r') as f:
        generatedTexts = json.load(f)

    return generatedTexts


def evaluate(args):
    timeStart = time.time()

    generatedTexts = readGeneratedTexts(args.input_file)[:args.n]
    numSamples = len(generatedTexts)

    customContext = args.context.strip()
    generatedTextsWithCustomContext = [customContext + (" " if len(customContext) > 0 else "") + g.strip()
                                       for g in generatedTexts]

    conceptSets, contexts = readEvaluationDatasetByName(args.dataset, numSamples, "")

    print("Loading perplexity model:", args.ppl_model_name)
    from VanillaGPT2 import VanillaGPT2
    pplModel = VanillaGPT2(args.ppl_model_name)
    tok = pplModel.tokenizer
    generatedTextsWithCustomContext = [tok.decode(tok.encode(g)[:32], skip_special_tokens=True)
                                       for g in generatedTextsWithCustomContext]

    res = []
    for sent_level in [args.sentence_level]:
        int_res = benchMetrics(generatedTextsWithCustomContext, contexts, pplModel, args.bs, conceptSets, args.do_coverage,
                               args.do_original_coverage, args.do_bleu, args.do_perplexity, args.do_semantic_coverage,
                               sent_level)
        res.append((sent_level, int_res))

    for sent_level, int_res in res:
        coverage, orgCoverage, meanPpl, selfbleu5, semCoverage = int_res
        print()
        print("#" * 50)
        print("Sentence Level:", sent_level)
        print(f"Coverage={coverage}, Mean Perplexity={meanPpl}, Self-BLEU-5={selfbleu5}")

    timeEnd = time.time()
    timeStr = datetime.timedelta(seconds=round(timeEnd - timeStart))
    print(f"Evaluation Time Elapsed for {numSamples} samples (h:mm:ss): {timeStr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=13374269,
                        help="The number of samples to evaluate on. Simply overshoot to evaluate on all available")
    parser.add_argument("--bs", type=int, default=256,
                        help="Batch size for perplexity calculations.")
    parser.add_argument("--ppl_model_name", type=str, default='gpt2-xl',
                        help="The model that should be used for perplexity calculations, e.g. gpt2-xl")
    parser.add_argument("--dataset", type=str, default="common_gen",
                        help=f"The dataset used for evaluation, i.e. one of {EVALUATION_DATASET_NAMES}")
    parser.add_argument("--input_file", type=str,
                        help="The path to the file with the generated texts. "
                             "Should be a file with a json list of strings.")
    parser.add_argument("--context", type=str, default="",
                        help="The starting context that was used to generate the texts, e.g. \"The\" or "
                             "\"The Sentence:\". This is only required if the texts were generated with a custom "
                             "context that is not included in the dataset.")
    parser.add_argument("--no_coverage", dest="do_coverage", action="store_false",
                        help="Use this to skip the coverage metric.")
    parser.add_argument("--no_bleu", dest="do_bleu", action="store_false",
                        help="Use this to skip the self-bleu-5 metric.")
    parser.add_argument("--no_perplexity", dest="do_perplexity", action="store_false",
                        help="Use this to skip the perplexity metric.")
    parser.add_argument("--sentence_level", dest="sentence_level", action="store_true",
                        help="Generated sentences will be stripped to first sentence before evaluation.")
    parser.set_defaults(do_coverage=True, do_original_coverage=True, do_bleu=True, do_perplexity=True,
                        do_semantic_coverage=True, sentence_level=False)

    args = parser.parse_args()

    assert args.n > 0
    assert args.bs > 0
    assert args.ppl_model_name in GPT2_MODELS
    args.dataset = args.dataset.lower()
    assert args.dataset in EVALUATION_DATASET_NAMES

    evaluate(args)

