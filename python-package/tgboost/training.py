import os


def run(file_training,
        file_validation,
        file_testing,
        file_output,
        categorical_features,
        early_stopping_rounds,
        maximize=True,
        eval_metric="auc",
        loss="logloss",
        eta=0.3,
        num_boost_round=1000,
        max_depth=6,
        scale_pos_weight=1,
        subsample=0.8,
        colsample=0.8,
        min_child_weight=1,
        min_sample_split=10,
        reg_lambda=1.0,
        gamma=0,
        num_thread=-1):

    if maximize:
        maximize = 'true'
    else:
        maximize = 'false'
    categorical_features = ",".join(categorical_features)

    path = os.path.dirname(os.path.realpath(__file__))
    command = "java -Xmx3600m -jar " + path + "/tgboost.jar" \
              + " " + file_training \
              + " " + file_validation \
              + " " + file_testing \
              + " " + file_output \
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

