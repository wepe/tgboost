from tree_node import TreeNode
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import copy_reg
import types
from time import time


# use copy_reg to make the instance method picklable,
# because multiprocessing must pickle things to sling them among process
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, _pickle_method)


class Tree(object):
    def __init__(self,
                 min_sample_split,
                 min_child_weight,
                 max_depth,
                 colsample,
                 rowsample,
                 reg_lambda,
                 gamma,
                 num_thread):
        self.root = None
        self.min_sample_split = min_sample_split
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.colsample = colsample
        self.rowsample = rowsample
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.alive_nodes = []
        self.name_to_node = {}
        # number of tree node of this tree
        self.nodes_cnt = 0
        # number of nan tree node of this tree
        # nan tree node is the third child of the tree node
        self.nan_nodes_cnt = 0

        if num_thread == -1:
            self.num_thread = cpu_count()
        else:
            self.num_thread = num_thread

        # to avoid divide zero
        self.reg_lambda = max(self.reg_lambda, 0.00001)

    def calculate_leaf_score(self, G, H):
        """
        According to xgboost, the leaf score is : - G / (H+lambda)
        """
        return - G / (H + self.reg_lambda)

    def calculate_split_gain(self, G_left, H_left, G_nan, H_nan, G_total, H_total):
        """
        According to xgboost, the scoring function is:
          gain = 0.5 * (GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)) - gamma

        this gain is the loss reduction, We want it to be as large as possible.

        """
        G_right = G_total - G_left - G_nan
        H_right = H_total - H_left - H_nan

        # if we let those with missing value go to a nan child
        gain_1 = 0.5 * (G_left**2/(H_left+self.reg_lambda)
                      + G_right**2/(H_right+self.reg_lambda)
                      + G_nan**2/(H_nan+self.reg_lambda)
                      - G_total**2/(H_total+self.reg_lambda)) - 2*self.gamma

        # if we let those with missing value go to left child
        gain_2 = 0.5 * ((G_left+G_nan) ** 2 / (H_left + H_nan + self.reg_lambda)
                        + G_right ** 2 / (H_right + self.reg_lambda)
                        - G_total ** 2 / (H_total + self.reg_lambda)) - self.gamma

        # if we let those with missing value go to right child
        gain_3 = 0.5 * ( G_left ** 2 / (H_left + self.reg_lambda)
                        + (G_right+G_nan) ** 2 / (H_right + H_nan + self.reg_lambda)
                        - G_total ** 2 / (H_total + self.reg_lambda)) - self.gamma

        nan_go_to = None
        gain = None
        if gain_1 == max([gain_1, gain_2, gain_3]):
            nan_go_to = 0  # nan child
            gain = gain_1
        elif gain_2 == max([gain_1, gain_2, gain_3]):
            nan_go_to = 1  # left child
            gain = gain_2
        else:
            nan_go_to = 2  # right child
            gain = gain_3

        # in this case, the trainset does not contains nan samples
        if H_nan == 0 and G_nan == 0:
            nan_go_to = 3

        return nan_go_to, gain

    def _process_one_attribute_list(self, class_list, (col_attribute_list, col_attribute_list_cutting_index, col)):
        """
        this function is base for parallel using multiprocessing,
        so all operation are read-only

        """
        ret = []
        # linear scan this column's attribute list, bin by bin
        for uint8_threshold in range(len(col_attribute_list_cutting_index) - 1):
            start_ind = col_attribute_list_cutting_index[uint8_threshold]
            end_ind = col_attribute_list_cutting_index[uint8_threshold + 1]
            inds = col_attribute_list["index"][start_ind:end_ind]
            tree_node_G_H = class_list.statistic_given_inds(inds)
            ret.append((col, uint8_threshold, tree_node_G_H))
        return ret

    def build(self, attribute_list, class_list, col_sampler, bin_structure):
        while len(self.alive_nodes) != 0:
            self.nodes_cnt += len(self.alive_nodes)
            # scan each selected attribute list
            attributes = []
            for col in col_sampler.col_selected:
                col_attribute_list = attribute_list[col]
                col_attribute_list_cutting_index = attribute_list.attribute_list_cutting_index[col]
                attributes.append((col_attribute_list, col_attribute_list_cutting_index, col))
            func = partial(self._process_one_attribute_list, class_list)
            pool = Pool(self.num_thread)
            rets = pool.map(func, attributes)
            pool.close()

            # for each attribute's ret
            for ret in rets:
                # for each threshold of this attribute
                for col, uint8_threshold, tree_node_G_H in ret:
                    # for each related tree node
                    for tree_node_name in tree_node_G_H.keys():
                        # get the original tree_node by tree_node_name using self.name_to_node
                        tree_node = self.name_to_node[tree_node_name]

                        G, H = tree_node_G_H[tree_node_name]
                        G_left, H_left = tree_node.get_Gleft_Hleft(col, G, H)
                        G_total, H_total = tree_node.Grad, tree_node.Hess
                        G_nan, H_nan = tree_node.Grad_missing[col], tree_node.Hess_missing[col]

                        nan_go_to, gain = self.calculate_split_gain(G_left, H_left, G_nan, H_nan, G_total, H_total)
                        tree_node.update_best_gain(col, uint8_threshold, bin_structure[col][uint8_threshold], gain, nan_go_to)

            # once had scan all column, we can get the best (feature,threshold,gain) for each alive tree node
            cur_level_node_size = len(self.alive_nodes)
            new_tree_nodes = []
            treenode_leftinds_naninds = []
            for _ in range(cur_level_node_size):
                # for each current alive node, get its best splitting
                tree_node = self.alive_nodes.pop(0)
                best_feature, best_uint8_threshold, best_threshold, best_gain, best_nan_go_to = tree_node.get_best_feature_threshold_gain()
                tree_node.nan_go_to = best_nan_go_to

                if best_gain > 0:
                    left_child = TreeNode(name=3*tree_node.name-1, depth=tree_node.depth+1, feature_dim=attribute_list.feature_dim)
                    right_child = TreeNode(name=3*tree_node.name+1, depth=tree_node.depth+1, feature_dim=attribute_list.feature_dim)
                    nan_child = None
                    # this case we can create the nan child
                    if best_nan_go_to == 0:
                        nan_child = TreeNode(name=3*tree_node.name, depth=tree_node.depth+1, feature_dim=attribute_list.feature_dim)
                        self.nan_nodes_cnt += 1
                    # this tree node is internal node
                    tree_node.internal_node_setter(best_feature, best_uint8_threshold, best_threshold, nan_child, left_child, right_child)

                    new_tree_nodes.append(left_child)
                    new_tree_nodes.append(right_child)
                    self.name_to_node[left_child.name] = left_child
                    self.name_to_node[right_child.name] = right_child
                    if nan_child is not None:
                        new_tree_nodes.append(nan_child)
                        self.name_to_node[nan_child.name] = nan_child

                    # to update class_list.corresponding_tree_node one pass,
                    # we should save (tree_node,left_inds, nan_inds)
                    left_inds = attribute_list[best_feature]["index"][0:attribute_list.attribute_list_cutting_index[best_feature][best_uint8_threshold+1]]
                    nan_inds = attribute_list.missing_value_attribute_list[best_feature]
                    treenode_leftinds_naninds.append((tree_node, (set(left_inds), set(nan_inds), best_nan_go_to)))
                else:
                    # this tree node is leaf node
                    leaf_score = self.calculate_leaf_score(tree_node.Grad, tree_node.Hess)
                    tree_node.leaf_node_setter(leaf_score)

            # update class_list.corresponding_tree_node one pass

            class_list.update_corresponding_tree_node(treenode_leftinds_naninds)

            # update histogram(Grad,Hess,num_sample) for each new tree node
            class_list.update_histogram_for_tree_node()

            # update Grad_missing, Hess_missing for each new tree node
            for tree_node in new_tree_nodes:
                tree_node.reset_Grad_Hess_missing()
            class_list.update_grad_hess_missing_for_tree_node(attribute_list.missing_value_attribute_list)

            # process the new tree nodes
            # satisfy max_depth? min_child_weight? min_sample_split?
            # if yes, it is leaf node, calculate its leaf score
            # if no, put into self.alive_node
            while len(new_tree_nodes) != 0:
                tree_node = new_tree_nodes.pop()
                if tree_node.depth >= self.max_depth \
                        or tree_node.Hess < self.min_child_weight \
                        or tree_node.num_sample <= self.min_sample_split:
                    tree_node.leaf_node_setter(self.calculate_leaf_score(tree_node.Grad, tree_node.Hess))
                    self.nodes_cnt += 1
                else:
                    self.alive_nodes.append(tree_node)

    def fit(self, attribute_list, class_list, row_sampler, col_sampler, bin_structure):
        # when we start to fit a tree, we first conduct row and column sampling
        col_sampler.shuffle()
        row_sampler.shuffle()
        class_list.sampling(row_sampler.row_mask)

        # then we create the root node, initialize histogram(Gradient sum and Hessian sum)
        root_node = TreeNode(name=1, depth=1, feature_dim=attribute_list.feature_dim)
        root_node.Grad_setter(class_list.grad.sum())
        root_node.Hess_setter(class_list.hess.sum())
        self.root = root_node

        # every time a new node is created, we put it into self.name_to_node
        self.name_to_node[root_node.name] = root_node

        # put it into the alive_node, and fill the class_list, all data are assigned to root node initially
        self.alive_nodes.append(root_node)
        for i in range(class_list.dataset_size):
            class_list.corresponding_tree_node[i] = root_node

        # then build the tree util there is no alive tree_node to split
        self.build(attribute_list, class_list, col_sampler, bin_structure)
        self.clean_up()

    def _predict(self, feature):
        """
        :param feature: feature of a single sample
        :return:
        """
        cur_tree_node = self.root
        while not cur_tree_node.is_leaf:
            # if the split feature's value of this sample is nan
            # then we first check whether cur_tree_node.nan_child exist
            # if exiist, then go to nan_child
            # if not exist, check cur_tree_node.nan_go_to.
            if np.isnan(feature[cur_tree_node.split_feature]):
                if cur_tree_node.nan_child is None:
                    if cur_tree_node.nan_go_to == 1:
                        cur_tree_node = cur_tree_node.left_child
                    elif cur_tree_node.nan_go_to == 2:
                        cur_tree_node = cur_tree_node.right_child
                    elif cur_tree_node.nan_go_to == 3:
                        # Sudden fantasy
                        # any other solution?
                        if cur_tree_node.left_child.num_sample > cur_tree_node.right_child.num_sample:
                            cur_tree_node = cur_tree_node.left_child
                        else:
                            cur_tree_node = cur_tree_node.right_child
                else:
                    cur_tree_node = cur_tree_node.nan_child
            elif feature[cur_tree_node.split_feature] <= cur_tree_node.split_threshold:
                cur_tree_node = cur_tree_node.left_child
            else:
                cur_tree_node = cur_tree_node.right_child
        return cur_tree_node.leaf_score

    def predict(self, features):
        pool = Pool(self.num_thread)
        preds = pool.map(self._predict, features)
        pool.close()
        return np.array(preds)

    def clean_up(self):
        del self.alive_nodes, self.min_sample_split, self.min_child_weight, self.rowsample,\
            self.colsample, self.max_depth, self.reg_lambda, self.gamma
