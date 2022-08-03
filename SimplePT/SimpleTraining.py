import SimplePromptTuningSetup, SimpleWordInclusionDataset
import transformers
import torch
import tqdm


def loadOptimizer(promptTuningSetup, finalLR, numWarmupSteps):
    optim = torch.optim.Adam(promptTuningSetup.parameters(), lr=finalLR, betas=(0.9, 0.999), eps=1e-8)
    return optim, transformers.optimization.get_constant_schedule_with_warmup(optim, numWarmupSteps)


def mainTrainingLoop(device, modelBase='gpt2'):
    batchSize, gradAccumulation = 1, 4
    maxSeqLen, maxPromptSize = 32, 32
    lr, stepsPerEpoch = 0.0001, 250
    numWarmupSteps = 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    model = SimplePromptTuningSetup.PromptTuningGPT2Setup(modelBase)

    model = model.to(device)
    model.prepareTrainingParameters()
    print("Number of training parameters:", len([p for p in model.parameters() if p.requires_grad]))

    trainDataloader = SimpleWordInclusionDataset.loadDummySingleSentenceDataset(tokenizer, batchSize, maxSeqLen,
                                                                               maxPromptSize)
    optimizer, optimSchedule = loadOptimizer(model, lr, numWarmupSteps)

    stepCounter = 0
    while True:
        trainLoss = []
        for x, y in tqdm.tqdm(trainDataloader):
            with torch.set_grad_enabled(True):
                txtIDs, pIDs, pAtt = [d.to(device) for d in x]
                maskedLabels = y.to(device)

                output = model.forward(txtIDs, pIDs, pAtt, maskedLabels=maskedLabels)

                lossOutputs = output.loss.cpu()
                loss = torch.mean(lossOutputs)
                loss = loss / gradAccumulation
                loss.backward()

                trainLoss.append(loss.detach().item())
                if ((stepCounter + 1) % gradAccumulation == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    optimSchedule.step()

                stepCounter += 1


if __name__ == '__main__':
    mainTrainingLoop('cuda:0')
