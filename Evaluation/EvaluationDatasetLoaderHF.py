from datasets import load_dataset

# Used to assert that dataset argument is correct
EVALUATION_DATASET_NAMES = ["common_gen", "c2gen"]


def readEvaluationDatasetByName(datasetName, numSamples, customContext):
    if datasetName == "common_gen":
        dataset = load_dataset('common_gen', split='test')
        conceptSets = dataset['concepts'][:numSamples]
        contexts = [customContext] * len(conceptSets)

    else:  # c2gen
        dataset = load_dataset('Non-Residual-Prompting/C2Gen', split='test')
        contexts, conceptSets = dataset['context'][:numSamples], dataset['keywords'][:numSamples]
        contexts = [c if len(c) > 0 else customContext for c in contexts]

    return conceptSets[:numSamples], contexts[:numSamples]

