import os
#os.environ['THEANO_FLAGS'] = "device=cpu"
from train_stack_nn import *



def writeResultModel(modelname, perf, params, other):
    loc = "C:/Users/jenazad/PycharmProjects/Regression-example/"
    f = open(loc+'stack_result.txt','a')
    f.write(modelname+ '\n')
    f.write(other+ '\n')
    f.write(str(perf)+ 'MAE \n')
    f.write(str(params)+ '\n')
    f.write('\n')
    f.close()


#


#

modelname = "ffnn_stack_1"
params = {'BN': True, 'dropout_rate': 0.2, 'optimizer': 'adam',
         'init': 'glorot_uniform', 'depth': 1, 'lr': 0.01, 'regl1': 1e-6, 'nb_neurone': 20}
perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead="stack")
other = "nbags=1, randp=False, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)

#

modelname = "ffnn_stack_2"
params = {'BN': True, 'dropout_rate': 0.3, 'optimizer': 'adam',
         'init': 'glorot_uniform', 'depth': 2, 'lr': 0.01, 'regl1': 1e-6, 'nb_neurone': 40}
perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead="stack")
other = "nbags=1, randp=False, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)

#

modelname = "ffnn_stack_3"
params = {'BN': True, 'dropout_rate': 0.4, 'optimizer': 'adam',
         'init': 'glorot_uniform', 'depth': 3, 'lr': 0.01, 'regl1': 1e-6, 'nb_neurone': 100}
perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead="stack")
other = "nbags=1, randp=False, varselect=False"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)