import numpy as np
from loss import SquareLoss, LogisticLoss, CustomizeLoss
from tree import Tree
from metric import get_metric
from attribute_list import AttributeList
from class_list import ClassList
from binning import BinStructure
from sampling import RowSampler, ColumnSampler
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s', datefmt="[%H:%M:%S]")


class TGBoost(object):
    """
    Tiny Gradient Boosting
    """
    def __init__(self):
        self.trees = []
        self.eta = None
        self.num_boost_round = None
        self.first_round_pred = None
        self.loss = None
        self.max_depth = None
        self.subsample = None
        self.colsample = None
        self.reg_lambda = None
        self.min_sample_split = None
        self.gamma = None
        self.num_thread = None
        self.min_child_weight = None
        self.scale_pos_weight = None
        self.eval_metric = None

    def fit(self,
            features,
            label,
            validation_data=(None, None),
            early_stopping_rounds=np.inf,
            maximize=True,
            eval_metric=None,
            loss="logisticloss",
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

        """
        :param features: np.array
        :param label: np.array
        :param eta: learning rate
        :param num_boost_round: number of boosting round
        :param max_depth: max depth of each tree
        :param subsample: row sample rate when building a tree
        :param colsample: column sample rate when building a tree
        :param min_sample_split: min number of samples in a leaf node
        :param loss: loss object
                     logisticloss,squareloss, or customize loss
        :param reg_lambda: lambda
        :param gamma: gamma
        :param num_thread: number of threself.tree_predict_Xad to parallel
        :param eval_metric: evaluation metric, provided: "accuracy"
        """
        self.eta = eta
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_sample_split = min_sample_split
        self.num_thread = num_thread
        self.eval_metric = eval_metric
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.first_round_pred = 0

        # initial loss function
        if loss == "logisticloss":
            self.loss = LogisticLoss()
        elif loss == "squareloss":
            self.loss = SquareLoss()
            self.first_round_pred = label.mean()
        else:
            try:
                self.loss = CustomizeLoss(loss)
            except:
                raise NotImplementedError("loss should be 'logisticloss','squareloss', or customize loss function")

        # initialize row_sampler, col_sampler, bin_structure, attribute_list, class_list
        row_sampler = RowSampler(features.shape[0], self.subsample)
        col_sampler = ColumnSampler(features.shape[1], self.colsample)
        bin_structure = BinStructure(features)
        attribute_list = AttributeList(features, bin_structure)
        class_list = ClassList(label)
        class_list.initialize_pred(self.first_round_pred)
        class_list.update_grad_hess(self.loss)

        # to evaluate on validation set and conduct early stopping
        # we should get (val_features,val_label)
        # and set some variable to check when to stop
        do_validation = True
        if not isinstance(validation_data, tuple):
            raise TypeError("validation_data should be (val_features, val_label)")

        val_features, val_label = validation_data
        val_pred = None
        if val_features is None or val_label is None:
            do_validation = False
        else:
            val_pred = np.ones(val_label.shape) * self.first_round_pred

        if maximize:
            best_val_metric = - np.inf
            best_round = 0
            become_worse_round = 0
        else:
            best_val_metric = np.inf
            best_round = 0
            become_worse_round = 0

        # start learning
        logging.info("TGBoost start training")
        for i in range(self.num_boost_round):
            # train current tree
            tree = Tree(self.min_sample_split,
                        self.min_child_weight,
                        self.max_depth,
                        self.colsample,
                        self.subsample,
                        self.reg_lambda,
                        self.gamma,
                        self.num_thread)
            tree.fit(attribute_list, class_list, row_sampler, col_sampler, bin_structure)

            # when finish building this tree, update the class_list.pred, grad, hess
            class_list.update_pred(self.eta)
            class_list.update_grad_hess(self.loss)

            # save this tree
            self.trees.append(tree)
            logging.debug("current tree has {} nodes, {} nan tree nodes".format(tree.nodes_cnt, tree.nan_nodes_cnt))

            # print training information
            if self.eval_metric is None:
                logging.info("TGBoost round {iteration}".format(iteration=i))
            else:
                try:
                    mertric_func = get_metric(self.eval_metric)
                except:
                    raise NotImplementedError("The given eval_metric is not provided")

                train_metric = mertric_func(self.loss.transform(class_list.pred), label)

                if not do_validation:
                    logging.info("TGBoost round {iteration}, train-{eval_metric}: {train_metric:.4f}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric))
                else:
                    val_pred += self.eta * tree.predict(val_features)
                    val_metric = mertric_func(self.loss.transform(val_pred), val_label)
                    logging.info("TGBoost round {iteration}, train-{eval_metric}: {train_metric:.4f}, val-{eval_metric}: {val_metric:.4f}".format(
                        iteration=i, eval_metric=self.eval_metric, train_metric=train_metric, val_metric=val_metric))

                    # check whether to early stop
                    if maximize:
                        if val_metric > best_val_metric:
                            best_val_metric = val_metric
                            best_round = i
                            become_worse_round = 0
                        else:
                            become_worse_round += 1
                        if become_worse_round > early_stopping_rounds:
                            logging.info("TGBoost training Stop, best round is {best_round}, best {eval_metric} is {best_val_metric:.4f}".format(
                                best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric))
                            break
                    else:
                        if val_metric < best_val_metric:
                            best_val_metric = val_metric
                            best_round = i
                            become_worse_round = 0
                        else:
                            become_worse_round += 1
                        if become_worse_round > early_stopping_rounds:
                            logging.info("TGBoost training Stop, best round is {best_round}, best val-{eval_metric} is {best_val_metric:.4f}".format(
                                best_round=best_round, eval_metric=eval_metric, best_val_metric=best_val_metric))
                            break

    def predict(self, features):
        assert len(self.trees) > 0
        preds = np.zeros((features.shape[0],))
        preds += self.first_round_pred
        for tree in self.trees:
            preds += self.eta * tree.predict(features)
        return self.loss.transform(preds)
