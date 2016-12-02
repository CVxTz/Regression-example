from train_stack_sk import *

from hyperopt import fmin, tpe, hp, Trials


def writeResultModel(modelname, perf, params, other):
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    f = open(loc+'opt_sk.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'MAE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()

def returnscore(model, modelname, data):

    def score(params):
        perf = trainModel(model, params=params, nbags=1, modelname = modelname, randp=False, shift=True, varselect=False, datasetRead="indata", data=data)
        writeResultModel(modelname, perf, params, "")
        print modelname, perf, params,

        return perf

    return score



def returnoptimize(space, score, max_evals):
    def optimize(trials):

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)


        print best

    return optimize




def optimizehyper(space, model, modelname, max_evals,data):

    score = returnscore(model, modelname, data)

    optimize = returnoptimize(space, score, max_evals)

    trials = Trials()

    optimize(trials)