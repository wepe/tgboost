import os
import random
import string
from .test import Model


def train(ftrain, fval, params={}):
    file_training = ftrain
    file_validation = fval
    categorical_features = params.get('categorical_features', [])
    early_stopping_rounds = params.get('early_stopping_rounds', 10)
    maximize = params.get('maximize', True)
    eval_metric = params.get('eval_metric', 'auc')
    loss = params.get('loss', 'logloss')
    eta = params.get('eta', 0.3)
    num_boost_round = params.get('num_boost_round', 50)
    max_depth = params.get('max_depth', 7)
    scale_pos_weight = params.get('scale_pos_weight', 1.)
    subsample = params.get('subsample', 0.8)
    colsample = params.get('colsample', 0.8)
    min_child_weight = params.get('min_child_weight', 1.)
    min_sample_split = params.get('min_sample_split', 10)
    reg_lambda = params.get('reg_lambda', 1.)
    gamma = params.get('gamma', 0.)
    num_thread = params.get('num_thread', -1)

    if maximize:
        maximize = 'true'
    else:
        maximize = 'false'
    categorical_features = ",".join(categorical_features)

    jar_path = os.path.dirname(os.path.realpath(__file__)) 
    file_model = '/tmp/' + '.tgboost-' + ''.join(random.sample(string.ascii_letters+string.digits, 20))

    command = "java -Xmx3600m -jar " + jar_path + "/tgboost.jar" \
              + " " + "training" \
              + " " + file_training \
              + " " + file_validation \
              + " " + file_model \
              + " " + str(early_stopping_rounds) \
              + " " + maximize \
              + " " + eval_metric \
              + " " + loss \
              + " " + str(eta) \
              + " " + str(num_boost_round) \
              + " " + str(max_depth) \
              + " " + str(scale_pos_weight) \
              + " " + str(subsample) \
              + " " + str(colsample) \
              + " " + str(min_child_weight) \
              + " " + str(min_sample_split) \
              + " " + str(reg_lambda) \
              + " " + str(gamma) \
              + " " + str(num_thread) \
              + " " + categorical_features

    os.system(command)
    return Model(file_model)
