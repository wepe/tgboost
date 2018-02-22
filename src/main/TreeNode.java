//        about TreeNode.index, an example:
//                    1
//           2        3       4
//        5  6  7   8 9 10  11 12 13
//
//        index of the root node is 1,
//        its left child's index is 3*root.index-1,
//        its right child's index is 3*root.index+1,
//        the middle child is nan_child, its index is 3*root.index

package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class TreeNode {
    public int index;
    public int depth;
    public int feature_dim;
    public boolean is_leaf;
    public int num_sample;
    //the gradient/hessian sum of the samples fall into this tree node
    public double Grad;
    public double Hess;
    //for split finding, record the gradient/hessian sum of the left
    public double[] G_left;
    public double[] H_left;
    //when split finding, record the best threshold, gain, missing value's branch for each feature
    private double[] best_thresholds;
    private double[] best_gains;
    private double[] best_nan_go_to;
    public double nan_go_to;
    //some data fall into this tree node
    //gradient sum, hessian sum of those with missing value for each feature
    public double[] Grad_missing;
    public double[] Hess_missing;
    //internal node
    public int split_feature;
    public double split_threshold;
    public ArrayList<Double> split_left_child_catvalue;
    public TreeNode nan_child;
    public TreeNode left_child;
    public TreeNode right_child;
    //leaf node
    double leaf_score;
    //for categorical feature,store (col,(value,(grad_sum,hess_sum)))
    public HashMap<Integer,HashMap<Integer,double[]>> cat_feature_col_value_GH = new HashMap<>();
    private HashMap<Integer,ArrayList<Integer>> cat_feature_col_leftcatvalue = new HashMap<>();

    public TreeNode(int index,double leaf_score){
        //leaf node construct
        this.is_leaf = true;
        this.index = index;
        this.leaf_score = leaf_score;
    }

    public TreeNode(int index,int split_feature,double split_threshold,double nan_go_to){
        //internal node construct,numeric split feature
        this.is_leaf = false;
        this.index = index;
        this.split_feature = split_feature;
        this.split_threshold = split_threshold;
        this.nan_go_to = nan_go_to;
    }

    public TreeNode(int index,int split_feature,ArrayList<Double> split_left_child_catvalue,double nan_go_to){
        //internal node construct,categorical split feature
        this.is_leaf = false;
        this.index = index;
        this.split_feature = split_feature;
        this.split_left_child_catvalue = split_left_child_catvalue;
        this.nan_go_to = nan_go_to;
    }


    public TreeNode(int index,int depth,int feature_dim,boolean is_leaf){
        this.index = index;
        this.depth = depth;
        this.feature_dim = feature_dim;
        this.is_leaf = is_leaf;
        this.G_left = new double[feature_dim];
        this.H_left = new double[feature_dim];
        this.best_thresholds = new double[feature_dim];
        this.best_gains = new double[feature_dim];
        this.best_nan_go_to = new double[feature_dim];
        this.Grad_missing = new double[feature_dim];
        this.Hess_missing = new double[feature_dim];

        Arrays.fill(this.best_gains,-Double.MAX_VALUE);

    }

    public void Grad_add(double value){
        Grad += value;
    }

    public void Hess_add(double value){
        Hess += value;
    }

    public void num_sample_add(double value){
        num_sample += value;
    }

    public void Grad_setter(double value){
        Grad = value;
    }

    public void Hess_setter(double value){
        Hess = value;
    }

    public void update_best_split(int col,double threshold,double gain,double nan_go_to){
        if(gain > best_gains[col]){
            best_gains[col] = gain;
            best_thresholds[col] = threshold;
            best_nan_go_to[col] = nan_go_to;
        }
    }

    public void set_categorical_feature_best_split(int col, ArrayList<Integer> left_child_catvalue,double gain,double nan_go_to){
        best_gains[col] = gain;
        best_nan_go_to[col] = nan_go_to;
        cat_feature_col_leftcatvalue.put(col,left_child_catvalue);
    }

    public ArrayList<Double> get_best_feature_threshold_gain(){
        int best_feature = 0;
        double max_gain = -Double.MAX_VALUE;
        for(int i=0;i<feature_dim;i++){
            if(best_gains[i]>max_gain){
                max_gain = best_gains[i];
                best_feature = i;
            }
        }
        //consider categorical feature
        ArrayList<Double> ret = new ArrayList<>();
        ret.add((double) best_feature);
        ret.add(max_gain);
        ret.add(best_nan_go_to[best_feature]);
        if(cat_feature_col_leftcatvalue.containsKey(best_feature)){
            for(double catvalue:cat_feature_col_leftcatvalue.get(best_feature)){
                ret.add(catvalue);
            }
        }else {
            ret.add(best_thresholds[best_feature]);
        }
        return ret;
    }

    public void internal_node_setter(double feature,double threshold,double nan_go_to,TreeNode nan_child,
                                     TreeNode left_child,TreeNode right_child,boolean is_leaf){
        this.split_feature = (int) feature;
        this.split_threshold = threshold;
        this.nan_go_to = nan_go_to;
        this.nan_child = nan_child;
        this.left_child = left_child;
        this.right_child = right_child;
        this.is_leaf = is_leaf;
        clean_up();
    }

    public void internal_node_setter(double feature,ArrayList<Double> left_child_catvalue,double nan_go_to,TreeNode nan_child,
                                     TreeNode left_child,TreeNode right_child,boolean is_leaf){
        this.split_feature = (int) feature;
        this.split_left_child_catvalue = left_child_catvalue;
        this.nan_go_to = nan_go_to;
        this.nan_child = nan_child;
        this.left_child = left_child;
        this.right_child = right_child;
        this.is_leaf = is_leaf;
        clean_up();
    }

    public void leaf_node_setter(double leaf_score,boolean is_leaf){
        this.is_leaf = is_leaf;
        this.leaf_score = leaf_score;
        clean_up();
    }

    private void clean_up(){
        //release memory
        best_thresholds = null;
        best_gains = null;
        best_nan_go_to = null;
        G_left = null;
        H_left = null;
        cat_feature_col_value_GH = null;
        cat_feature_col_leftcatvalue = null;
    }

}
