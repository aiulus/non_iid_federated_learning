import numpy as np
import random

def getDirichletProbs(alpha, nClients, nLabels):
    probs = [list(np.random.dirichlet(np.repeat(alpha, nClients))) for i in range(nLabels)]
    return probs

def partitionWithProbs(data,probs):
    nClients = len(probs[0])
    partitions = [[] for i in range(nClients)]
    cumulatives = []
    for labelProbs in probs:
        if not sum(labelProbs) == 1:
            # Normalize probabilities if not already
            labelProbs = [prob/sum(labelProbs) for prob in labelProbs]
        cumulatives.append([sum(labelProbs[0:i+1]) for i in range(len(labelProbs))])

    for point in data:
        y = random.random() 
        for client in range(nClients):
            if y <= cumulatives[point][client]:
                partitions[client].append(point)
                break

    return partitions


def partitionData(data,alpha,nClients):
    partitions = partitionWithProbs(data,getDirichletProbs(alpha,nClients,len(set(data))))
    return(partitions)