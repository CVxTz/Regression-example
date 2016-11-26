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


##
#
#modelname = "ffnn1"
#params = {'BN': True, 'dropout_rate': 0.2, 'optimizer': 'adadelta',
#         'init': 'he_normal', 'depth': 1, 'lr': 0.1, 'regl1': 0, 'nb_neurone': 200}
#perf = trainModelNn(params=params, nbags=3, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
##
#
#modelname = "ffnn2"
#params = {'BN': True, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.05, 'regl1': 1e-5, 'nb_neurone': 170}
#perf = trainModelNn(params=params, nbags=3, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
##
#
#modelname = "ffnn3"
#params = {'BN': True, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01, 'regl1': 1e-6, 'nb_neurone': 200}
#perf = trainModelNn(params=params, nbags=3, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)

#

# modelname = "ffnn4"
# params = {'BN': True, 'dropout_rate': 0.3, 'optimizer': 'rmsprop',
#          'init': 'he_normal', 'depth': 1, 'lr': 0.1, 'regl1': 0.0001, 'nb_neurone': 1000}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# #
#
# modelname = "ffnn5"
# params = {'BN': True, 'dropout_rate': 0.2, 'optimizer': 'rmsprop',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.1, 'regl1': 0.0001, 'nb_neurone': 1000}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# #
#
# modelname = "ffnn6"
# params = {'BN': True, 'dropout_rate': 0.4, 'optimizer': 'rmsprop',
#          'init': 'he_normal', 'depth': 3, 'lr': 0.1, 'regl1': 0.0001, 'nb_neurone': 1000}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# #
#
# modelname = "ffnn7"
# params = {'BN': True, 'dropout_rate': 0.5, 'optimizer': 'rmsprop',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.1, 'regl1': 0.01, 'nb_neurone': 1000}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)

#modelname = "ffnn8"
#params = {'BN': True, 'dropout_rate': 0.5, 'optimizer': 'rmsprop',
#         'init': 'he_normal', 'depth': 1, 'lr': 0.01, 'regl1': 0.0001, 'nb_neurone': 1000}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead = "comp")
#other = "nbags=1, randp=False, varselect=False, datasetRead = comp"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
##
#modelname = "ffnn9"
#params = {'BN': True, 'dropout_rate': 0.5, 'optimizer': 'rmsprop',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01, 'regl1': 0.00001, 'nb_neurone': 500}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead = "comp")
#other = "nbags=1, randp=False, varselect=False, datasetRead = comp"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
##
#modelname = "ffnn10"
#params = {'BN': True, 'dropout_rate': 0.5, 'optimizer': 'rmsprop',
#         'init': 'he_normal', 'depth': 3, 'lr': 0.01, 'regl1': 0.001, 'nb_neurone': 500}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, datasetRead = "comp")
#other = "nbags=1, randp=False, varselect=False, datasetRead = comp"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)

#modelname = "ffnn11"
#params = {'BN': False, 'dropout_rate': 0.2, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01, 'regl2': 0, 'regl1': 0, 'nb_neurone': 500}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)

#modelname = "ffnn12"
#params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0, 'regl1': 0, 'nb_neurone': 500}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelna#me = "ffnn13"
#params = {'BN': False, 'dropout_rate': 0.5, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0, 'regl1': 0.0001, 'nb_neurone': 500}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn14"
#params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0.001, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn15"
#params = {'BN': False, 'dropout_rate': 0.4, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 3, 'lr': 0.01,  'regl2': 0, 'regl1': 0.0001, 'nb_neurone': 500, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn16"
#params = {'BN': False, 'dropout_rate': 0.5, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0.01, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn17"
#params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0.001, 'regl1': 0, 'nb_neurone': 200, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn18"
#params = {'BN': False, 'dropout_rate': 0.5, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 3, 'lr': 0.01,  'regl2': 0.01, 'regl1': 0.0001, 'nb_neurone': 700, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn19"
#params = {'BN': False, 'dropout_rate': 0.5, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.01,  'regl2': 0.00001, 'regl1': 0.01, 'nb_neurone': 1000, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn20"
#params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.003,  'regl2': 0.001, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn21"
#params = {'BN': False, 'dropout_rate': 0.4, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 3, 'lr': 0.0001,  'regl2': 0.0001, 'regl1': 0.0001, 'nb_neurone': 500, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#modelname = "ffnn22"
#params = {'BN': False, 'dropout_rate': 0.5, 'optimizer': 'adam',
#         'init': 'he_normal', 'depth': 2, 'lr': 0.001,  'regl2': 0.01, 'regl1': 0, 'nb_neurone': 600, 'prelu':False}
#perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
#other = "nbags=1, randp=False, varselect=False"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)

# modelname = "ffnn23"
# params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.003,  'regl2': 0.01, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
# perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# modelname = "ffnn24"
# params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.003,  'regl2': 0.1, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
# perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
# modelname = "ffnn25"
# params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.003,  'regl2': 1, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
# perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
#
# modelname = "ffnn26"
# params = {'BN': False, 'dropout_rate': 0.3, 'optimizer': 'adam',
#          'init': 'he_normal', 'depth': 2, 'lr': 0.003,  'regl2': 10, 'regl1': 0, 'nb_neurone': 500, 'prelu':False}
# perf = trainModelNn(params=params, nbags=1, modelname = modelname, randp=False, varselect=False)
# other = "nbags=1, randp=False, varselect=False"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)


#modelname = "ffnn27"
#params = {}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, model2 = True)
#other = "nbags=1, randp=False, varselect=False, model2 = True"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)
#
#
#modelname = "ffnn28"
#params = {}
#perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, model2 = True, datasetRead = "impact")
#other = "nbags=1, randp=False, varselect=False, model2 = True, datasetRead = impact"
#
#print modelname, perf, params, other
#writeResultModel(modelname, perf, params, other)

#
# modelname = "ffnn29"
# params = {}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, model2 = True)
# other = "nbags=1, randp=False, varselect=False, model2 = True"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)
#
#
# modelname = "ffnn30"
# params = {}
# perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, model2 = True, datasetRead = "impact")
# other = "nbags=1, randp=False, varselect=False, model2 = True, datasetRead = impact"
#
# print modelname, perf, params, other
# writeResultModel(modelname, perf, params, other)


modelname = "ffnn31"
params = {}
perf = trainModelNn(params=params, nbags=5, modelname = modelname, randp=False, varselect=False, model2 = True, datasetRead = "impact")
other = "nbags=5, randp=False, varselect=False, model2 = True, datasetRead = impact"

print modelname, perf, params, other
writeResultModel(modelname, perf, params, other)