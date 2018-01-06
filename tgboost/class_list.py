# design a ClassList,store (label,leaf node,prediction,grad,hess) ordered by data index

import numpy as np


class ClassList(object):
    def __init__(self, label):
        self.dataset_size = label.shape[0]
        self.label = label
        self.corresponding_tree_node = [None for _ in range(self.dataset_size)]
        self.pred = np.ones(label.shape, dtype="float32")
        self.grad = np.empty(label.shape, dtype="float32")
        self.hess = np.empty(label.shape, dtype="float32")

    def initialize_pred(self, first_round_pred):
        self.pred *= first_round_pred

    def update_pred(self, eta):
        for i in range(self.dataset_size):
            self.pred[i] += eta*self.corresponding_tree_node[i].leaf_score

    def update_grad_hess(self, loss):
        # self.grad = loss.grad(self.pred, self.label)
        # self.hess = loss.hess(self.pred, self.label)
        self.grad[:] = loss.grad(self.pred, self.label)
        self.hess[:] = loss.hess(self.pred, self.label)

    def sampling(self, row_mask):
        self.grad *= row_mask
        self.hess *= row_mask

    def statistic_given_inds(self, inds):
        # scan the given the index, calculate each alive tree node's (G,H)
        ret = {}
        for i in inds:
            tree_node = self.corresponding_tree_node[i]
            if tree_node.is_leaf:
                continue
            else:
                if tree_node.name not in ret:
                    ret[tree_node.name] = [0., 0.]
                ret[tree_node.name][0] += self.grad[i]
                ret[tree_node.name][1] += self.hess[i]
        return ret

    def update_corresponding_tree_node(self, treenode_leftinds_naninds):
        # scan the class list, if the data fall into tree_node
        # then we see whether its index is in left_inds or right_inds, update the corresponding tree node
        map = dict(treenode_leftinds_naninds)
        for i in range(self.dataset_size):
            tree_node = self.corresponding_tree_node[i]
            if not tree_node.is_leaf:
                left_inds = map[tree_node][0]
                nan_inds = map[tree_node][1]
                nan_go_to = map[tree_node][2]
                if i in left_inds:
                    self.corresponding_tree_node[i] = tree_node.left_child
                elif i in nan_inds:
                    if nan_go_to == 0:
                        self.corresponding_tree_node[i] = tree_node.nan_child
                    elif nan_go_to == 1:
                        self.corresponding_tree_node[i] = tree_node.left_child
                    else:
                        self.corresponding_tree_node[i] = tree_node.right_child
                else:
                    self.corresponding_tree_node[i] = tree_node.right_child

    def update_histogram_for_tree_node(self):
        # scan the class list
        # update histogram(Grad,Hess,num_sample) for each alive(new) tree node
        for i in range(self.dataset_size):
            tree_node = self.corresponding_tree_node[i]
            if not tree_node.is_leaf:
                tree_node.Grad_add(self.grad[i])
                tree_node.Hess_add(self.hess[i])
                tree_node.num_sample_add(1)

    def update_grad_hess_missing_for_tree_node(self, missing_value_attribute_list):
        for col in range(len(missing_value_attribute_list)):
            for i in missing_value_attribute_list[col]:
                tree_node = self.corresponding_tree_node[i]
                if not tree_node.is_leaf:
                    tree_node.Grad_missing[col] += self.grad[i]
                    tree_node.Hess_missing[col] += self.hess[i]
