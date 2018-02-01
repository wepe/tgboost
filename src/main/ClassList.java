package main;

import org.omg.CORBA.DATA_CONVERSION;

import java.util.ArrayList;
import java.util.Arrays;

public class ClassList {
    public int dataset_size;
    public double[] label;
    public TreeNode[] corresponding_tree_node;
    public double[] pred;
    public double[] grad;
    public double[] hess;

    public ClassList(TrainData data){
        this.dataset_size = data.dataset_size;
        this.label = data.label;
        this.pred = new double[dataset_size];
        this.grad = new double[dataset_size];
        this.hess = new double[dataset_size];
        this.corresponding_tree_node = new TreeNode[dataset_size];
    }

    public void initialize_pred(double first_round_pred){
        Arrays.fill(pred, first_round_pred);
    }

    public void update_pred(double eta){
        for(int i=0;i<dataset_size;i++){
            pred[i] += eta * corresponding_tree_node[i].leaf_score;
        }
    }

    public void update_grad_hess(Loss loss,double scale_pos_weight){
        grad = loss.grad(pred,label);
        hess = loss.hess(pred,label);
        if(scale_pos_weight != 1.0){
            for(int i=0;i<dataset_size;i++){
                if(label[i]==1){
                    grad[i] *= scale_pos_weight;
                    hess[i] *= scale_pos_weight;
                }
            }
        }
    }

    public void sampling(ArrayList<Double> row_mask){
        for(int i=0;i<dataset_size;i++){
            grad[i] *= row_mask.get(i);
            hess[i] *= row_mask.get(i);
        }
    }

    //TODO
    //parallel each col's calculation
    public void update_grad_hess_missing_for_tree_node(int[][] missing_value_attribute_list){
        for(int col=0;col<missing_value_attribute_list.length;col++){
            for(int i:missing_value_attribute_list[col]){
                TreeNode treenode = corresponding_tree_node[i];
                if(!treenode.is_leaf){
                    treenode.Grad_missing[col] += grad[i];
                    treenode.Hess_missing[col] += hess[i];
                }
            }
        }
    }

    public void update_Grad_Hess_numsample_for_tree_node(){
        for(int i=0;i<dataset_size;i++){
            TreeNode treenode = corresponding_tree_node[i];
            if(!treenode.is_leaf){
                treenode.Grad_add(grad[i]);
                treenode.Hess_add(hess[i]);
                treenode.num_sample_add(1);
            }
        }
    }

    public void update_corresponding_tree_node(AttributeList attribute_list){
        for(int i=0;i<dataset_size;i++){
            TreeNode treenode = corresponding_tree_node[i];
            if(!treenode.is_leaf){
                int split_feature = treenode.split_feature;
                double nan_go_to = treenode.nan_go_to;
                double val = attribute_list.origin_feature[i][split_feature];
                //consider categorical feature
                if(attribute_list.cat_features_cols.contains(split_feature)){
                    ArrayList<Double> left_child_catvalue = treenode.split_left_child_catvalue;
                    if(val==Data.NULL){
                        if(nan_go_to==0){
                            corresponding_tree_node[i] = treenode.nan_child;
                        }else if(nan_go_to==1){
                            corresponding_tree_node[i] = treenode.left_child;
                        }else {
                            corresponding_tree_node[i] = treenode.right_child;
                        }
                    }else if(left_child_catvalue.contains(val)){
                        corresponding_tree_node[i] = treenode.left_child;
                    }else {
                        corresponding_tree_node[i] = treenode.right_child;
                    }
                }else {
                    double split_threshold = treenode.split_threshold;
                    if(val== Data.NULL){
                        if(nan_go_to==0){
                            corresponding_tree_node[i] = treenode.nan_child;
                        }else if(nan_go_to==1){
                            corresponding_tree_node[i] = treenode.left_child;
                        }else {
                            corresponding_tree_node[i] = treenode.right_child;
                        }
                    }else if(val<=split_threshold){
                        corresponding_tree_node[i] = treenode.left_child;
                    }else {
                        corresponding_tree_node[i] = treenode.right_child;
                    }
                }
            }
        }
    }

}
