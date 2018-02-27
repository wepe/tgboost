package main;

import java.util.*;
import java.util.concurrent.*;

public class Tree {
    private TreeNode root;
    private int min_sample_split;
    private double min_child_weight;
    private int max_depth;
    private double colsample;
    private double rowsample;
    private double lambda;
    private double gamma;
    private int num_thread;
    private ArrayList<Integer> cat_features_cols;
    private Queue<TreeNode> alive_nodes = new LinkedList<>();
    //number of tree node of this tree
    public int nodes_cnt = 0;
    //number of nan tree node of this tree
    public int nan_nodes_cnt = 0;

    public Tree(TreeNode root){
        this.root = root;
        this.num_thread = Runtime.getRuntime().availableProcessors();
    }

    public Tree(int min_sample_split,
                double min_child_weight,
                int max_depth,
                double colsample,
                double rowsample,
                double lambda,
                double gamma,
                int num_thread,
                ArrayList<Integer> cat_features_cols){
        this.min_sample_split = min_sample_split;
        this.min_child_weight = min_child_weight;
        this.max_depth = max_depth;
        this.colsample = colsample;
        this.rowsample = rowsample;
        this.lambda = lambda;
        this.gamma = gamma;
        this.cat_features_cols = cat_features_cols;

        if(num_thread==-1){
            this.num_thread = Runtime.getRuntime().availableProcessors();
        }else {
            this.num_thread = num_thread;
        }
        //to avoid divide zero
        this.lambda = Math.max(this.lambda, 0.00001);
    }

    private double calculate_leaf_score(double G,double H){
        //According to xgboost, the leaf score is : - G / (H+lambda)
        return -G/(H+this.lambda);
    }

    private double[] calculate_split_gain(double G_left,double H_left,double G_nan,double H_nan,double G_total,double H_total){
        //According to xgboost, the scoring function is:
        //     gain = 0.5 * (GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)) - gamma
        //this gain is the loss reduction, We want it to be as large as possible.
        double G_right = G_total - G_left - G_nan;
        double H_right = H_total - H_left - H_nan;

        //if we let those with missing value go to a nan child
        double gain_1 = 0.5 * (
                Math.pow(G_left,2)/(H_left+lambda)
                        + Math.pow(G_right,2)/(H_right+lambda)
                        + Math.pow(G_nan,2)/(H_nan+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda))-gamma;

        //uncomment this line, then we use xgboost's method to deal with missing value
        //gain_1 = -Double.MAX_VALUE;

        //if we let those with missing value go to left child
        double gain_2 = 0.5 * (
                Math.pow(G_left+G_nan,2)/(H_left+H_nan+lambda)
                        + Math.pow(G_right,2)/(H_right+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda))-gamma;

        //if we let those with missing value go to right child
        double gain_3 = 0.5 * (
                Math.pow(G_left,2)/(H_left+lambda)
                        + Math.pow(G_right+G_nan,2)/(H_right+H_nan+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda))-gamma;

        double nan_go_to;
        double gain = Math.max(gain_1,Math.max(gain_2,gain_3));
        if(gain_1==gain){
            nan_go_to = 0; //nan child
        }else if(gain_2==gain){
            nan_go_to = 1; //left child
        }else{
            nan_go_to = 2; //right child
        }

        //in this case, the trainset does not contains nan samples
        if(H_nan==0 && G_nan==0){
            nan_go_to = 3;
        }

        return new double[]{nan_go_to,gain};
    }

    public void fit(AttributeList attribute_list,
                    ClassList class_list,
                    RowSampler row_sampler,
                    ColumnSampler col_sampler){
        //when we start to fit a tree, we first conduct row and column sampling
        col_sampler.shuffle();
        row_sampler.shuffle();
        class_list.sampling(row_sampler.row_mask);


        //then we create the root node, initialize histogram(Gradient sum and Hessian sum)
        TreeNode root_node = new TreeNode(1,1,attribute_list.feature_dim,false);
        root_node.Grad_setter(sum(class_list.grad));
        root_node.Hess_setter(sum(class_list.hess));
        this.root = root_node;


        //put it into the alive_node, and fill the class_list, all data are assigned to root node initially
        alive_nodes.offer(root_node);


        for(int i=0;i<class_list.dataset_size;i++){
            class_list.corresponding_tree_node[i] = root_node;
        }

        //update Grad_missing Hess_missing for root node
        class_list.update_grad_hess_missing_for_tree_node(attribute_list.missing_value_attribute_list);

        //then build the tree util there is no alive tree_node to split
        build(attribute_list,class_list,col_sampler);
        clean_up();
    }

    class ProcessEachNumericFeature implements Runnable{
        public int col;
        public AttributeList attribute_list;
        public ClassList class_list;
        public ProcessEachNumericFeature(int col,AttributeList attribute_list,ClassList class_list){
            this.col = col;
            this.attribute_list = attribute_list;
            this.class_list = class_list;
        }

        @Override
        public void run(){
            for(int interval=0;interval<attribute_list.cutting_inds[col].length-1;interval++){
                //update the corresponding treenode's G_left,H_left with this inds's sample
                int[] inds = attribute_list.cutting_inds[col][interval];

                HashSet<TreeNode> nodes = new HashSet<>();
                for(int ind:inds){
                    TreeNode treenode = class_list.corresponding_tree_node[ind];
                    if(treenode.is_leaf) continue;

                    nodes.add(treenode);
                    treenode.G_left[col] += class_list.grad[ind];
                    treenode.H_left[col] += class_list.hess[ind];
                }
                //update each treenode's best split using this feature
                for(TreeNode node:nodes){
                    double G_left = node.G_left[col];
                    double H_left = node.H_left[col];
                    double G_total = node.Grad;
                    double H_total = node.Hess;
                    double G_nan = node.Grad_missing[col];
                    double H_nan = node.Hess_missing[col];
                    double[] ret = calculate_split_gain(G_left,H_left,G_nan,H_nan,G_total,H_total);
                    double nan_go_to = ret[0];
                    double gain = ret[1];
                    node.update_best_split(col,attribute_list.cutting_thresholds[col][interval],gain,nan_go_to);
                }
            }

        }
    }

    class ProcessEachCategoricalFeature implements Runnable{
        public int col;
        public AttributeList attribute_list;
        public ClassList class_list;

        public ProcessEachCategoricalFeature(int col,AttributeList attribute_list,ClassList class_list){
            this.col = col;
            this.attribute_list = attribute_list;
            this.class_list = class_list;
        }

        @Override
        public void run(){
            HashSet<TreeNode> nodes = new HashSet<>();
            for(int interval=0;interval<attribute_list.cutting_inds[col].length;interval++){
                //update the corresponding treenode's cat_feature_col_value_GH
                int[] inds = attribute_list.cutting_inds[col][interval];
                int cat_value = (int) attribute_list.cutting_thresholds[col][interval];
                for(int ind:inds){
                    TreeNode treenode = class_list.corresponding_tree_node[ind];
                    if(treenode.is_leaf) continue;

                    if(!nodes.contains(treenode)){
                        nodes.add(treenode);
                        treenode.cat_feature_col_value_GH.put(col,new HashMap<>());
                    }

                    if(treenode.cat_feature_col_value_GH.get(col).containsKey(cat_value)){
                        treenode.cat_feature_col_value_GH.get(col).get(cat_value)[0] += class_list.grad[ind];
                        treenode.cat_feature_col_value_GH.get(col).get(cat_value)[1] += class_list.hess[ind];
                    }else {
                        treenode.cat_feature_col_value_GH
                                .get(col).put(cat_value,new double[]{class_list.grad[ind],class_list.hess[ind]});
                    }
                }
            }

            for(TreeNode node:nodes){
                double[][] catvalue_GdivH = new double[node.cat_feature_col_value_GH.get(col).size()][4];
                int i=0;
                for(int catvalue:node.cat_feature_col_value_GH.get(col).keySet()){
                    catvalue_GdivH[i][0] = catvalue;
                    catvalue_GdivH[i][1] = node.cat_feature_col_value_GH.get(col).get(catvalue)[0];
                    catvalue_GdivH[i][2] = node.cat_feature_col_value_GH.get(col).get(catvalue)[1];
                    catvalue_GdivH[i][3] = catvalue_GdivH[i][1] / catvalue_GdivH[i][2];
                    i++;
                }
                Arrays.sort(catvalue_GdivH, new Comparator<double[]>() {
                    @Override
                    public int compare(double[] a, double[] b) {
                        return Double.compare(a[3],b[3]);
                    }
                });

                double G_total = node.Grad;
                double H_total = node.Hess;
                double G_nan = node.Grad_missing[col];
                double H_nan = node.Hess_missing[col];
                double G_left = 0;
                double H_left = 0;
                int best_split = -1;
                double best_gain = -Double.MAX_VALUE;
                double best_nan_go_to = -1;
                for(i=0;i<catvalue_GdivH.length;i++){
                    G_left += catvalue_GdivH[i][1];
                    H_left += catvalue_GdivH[i][2];
                    double[] ret = calculate_split_gain(G_left,H_left,G_nan,H_nan,G_total,H_total);
                    double nan_go_to = ret[0];
                    double gain = ret[1];
                    if(gain > best_gain){
                        best_gain = gain;
                        best_split = i;
                        best_nan_go_to = nan_go_to;
                    }
                }
                ArrayList<Integer> left_child_catvalue = new ArrayList<>();
                for(i=0;i<=best_split;i++){
                    left_child_catvalue.add((int) catvalue_GdivH[i][0]);
                }
                node.set_categorical_feature_best_split(col,left_child_catvalue,best_gain,best_nan_go_to);
            }

        }

    }


    private void build(AttributeList attribute_list,
                      ClassList class_list,
                      ColumnSampler col_sampler){
        while(!alive_nodes.isEmpty()){
            nodes_cnt += alive_nodes.size();

            //parallelly scan and process each selected attribute list
            ExecutorService pool = Executors.newFixedThreadPool(num_thread);
            for(int col:col_sampler.col_selected){
                if(attribute_list.cat_features_cols.contains(col)){
                    pool.execute(new ProcessEachCategoricalFeature(col,attribute_list,class_list));
                }else{
                    pool.execute(new ProcessEachNumericFeature(col,attribute_list,class_list));
                }
            }

            pool.shutdown();
            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            //once had scan all column, we can get the best (feature,threshold,gain) for each alive tree node
            int cur_level_node_size = alive_nodes.size();
            Queue<TreeNode> new_tree_nodes = new LinkedList<>();

            //time consumption: 0.0x ms
            for(int i=0;i<cur_level_node_size;i++){
                //pop each alive treenode
                TreeNode treenode = alive_nodes.poll();

                //consider categorical feature
                ArrayList<Double> ret = treenode.get_best_feature_threshold_gain();
                double best_feature = ret.get(0);
                double best_gain = ret.get(1);
                double best_nan_go_to = ret.get(2);
                double best_threshold = 0;
                ArrayList<Double> left_child_catvalue = new ArrayList<>();
                if(cat_features_cols.contains((int) best_feature)){
                    for(int j=3;j<ret.size();j++){
                        left_child_catvalue.add(ret.get(j));
                    }
                }else {
                    best_threshold = ret.get(3);
                }

                if(best_gain<=0){
                    //this node is leaf node
                    double leaf_score = calculate_leaf_score(treenode.Grad,treenode.Hess);
                    treenode.leaf_node_setter(leaf_score,true);
                }else {
                    //this node is internal node
                    TreeNode left_child = new TreeNode(3*treenode.index-1,treenode.depth+1,treenode.feature_dim,false);
                    TreeNode right_child = new TreeNode(3*treenode.index+1,treenode.depth+1,treenode.feature_dim,false);
                    TreeNode nan_child = null;
                    if(best_nan_go_to==0){
                        //this case we create the nan child
                        nan_child = new TreeNode(3*treenode.index,treenode.depth+1,treenode.feature_dim,false);
                        nan_nodes_cnt+=1;
                    }
                    //consider categorical feature
                    if(cat_features_cols.contains((int) best_feature)){
                        treenode.internal_node_setter(best_feature,left_child_catvalue,best_nan_go_to,nan_child,left_child,right_child,false);
                    }else {
                        treenode.internal_node_setter(best_feature,best_threshold,best_nan_go_to,nan_child,left_child,right_child,false);
                    }

                    new_tree_nodes.offer(left_child);
                    new_tree_nodes.offer(right_child);
                    if(nan_child != null){
                        new_tree_nodes.offer(nan_child);
                    }
                }
            }

            //update class_list.corresponding_tree_node
            class_list.update_corresponding_tree_node(attribute_list);

            //update (Grad,Hess,num_sample) for each new tree node
            class_list.update_Grad_Hess_numsample_for_tree_node();

            //update Grad_missing, Hess_missing for each new tree node
            //time consumption: 5ms
            class_list.update_grad_hess_missing_for_tree_node(attribute_list.missing_value_attribute_list);

            //process the new tree nodes
            //satisfy max_depth? min_child_weight? min_sample_split?
            //if yes, it is leaf node, calculate its leaf score
            //if no, put into self.alive_node
            while(new_tree_nodes.size()!=0){
                TreeNode treenode = new_tree_nodes.poll();
                if(treenode.depth>=max_depth || treenode.Hess<min_child_weight || treenode.num_sample<=min_sample_split){
                    treenode.leaf_node_setter(calculate_leaf_score(treenode.Grad,treenode.Hess),true);
                }else {
                    alive_nodes.offer(treenode);
                }
            }
        }
    }


    class PredictCallable implements Callable{
        private float[] feature;
        public PredictCallable(float[] feature){
            this.feature = feature;
        }
        @Override
        public Double call(){
            TreeNode cur_tree_node = root;
            while(!cur_tree_node.is_leaf){
                if(feature[cur_tree_node.split_feature]==Data.NULL){
                    //it is missing value
                    if(cur_tree_node.nan_go_to==0){
                        cur_tree_node = cur_tree_node.nan_child;
                    }else if(cur_tree_node.nan_go_to==1){
                        cur_tree_node = cur_tree_node.left_child;
                    }else if(cur_tree_node.nan_go_to==2){
                        cur_tree_node = cur_tree_node.right_child;
                    }else {
                        //trainset has not missing value for this feature,
                        // so we should decide which branch the testset's missing value go to
                        if(cur_tree_node.left_child.num_sample>cur_tree_node.right_child.num_sample){
                            cur_tree_node = cur_tree_node.left_child;
                        }else {
                            cur_tree_node = cur_tree_node.right_child;
                        }
                    }

                }else{
                    //not missing value
                    // consider split_feature categorical or numeric
                    if(cur_tree_node.split_left_child_catvalue!=null){
                        if(cur_tree_node.split_left_child_catvalue.contains((double) feature[cur_tree_node.split_feature])){
                            cur_tree_node = cur_tree_node.left_child;
                        }else {
                            cur_tree_node = cur_tree_node.right_child;
                        }
                    }else {
                        if(feature[cur_tree_node.split_feature]<=cur_tree_node.split_threshold){
                            cur_tree_node = cur_tree_node.left_child;
                        }else {
                            cur_tree_node = cur_tree_node.right_child;
                        }
                    }
                }
            }
            return cur_tree_node.leaf_score;
        }
    }


    public double[] predict(float[][] features){
        ExecutorService pool = Executors.newFixedThreadPool(num_thread);
        List<Future> list = new ArrayList<Future>();
        for(int i=0;i<features.length;i++){
            Callable c = new PredictCallable(features[i]);
            Future f = pool.submit(c);
            list.add(f);
        }

        pool.shutdown();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        double[] ret = new double[features.length];
        for(int i=0;i<ret.length;i++){
            try{
                ret[i] = (double) list.get(i).get();
            }catch (InterruptedException e){
                e.printStackTrace();
            }catch (ExecutionException e){
                e.printStackTrace();
            }
        }
        return ret;
    }

    private void clean_up(){
        this.alive_nodes = null;
    }

    private double sum(double[] vals){
        double s = 0;
        for(double v:vals) s+=v;
        return s;
    }

    public TreeNode getRoot() {
        return root;
    }
}

