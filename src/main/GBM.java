package main;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;


public class GBM {
    private ArrayList<Tree> trees = new ArrayList<>();
    private double eta;
    private int num_boost_round;
    private double first_round_pred;
    private Loss loss;
    private int max_depth;
    private double rowsample;
    private double colsample;
    private double lambda;
    private int min_sample_split;
    private double gamma;
    private int num_thread;
    private double min_child_weight;
    private double scale_pos_weight;
    private String eval_metric;

    private static Logger logger = Logger.getLogger("InfoLogging");

    public GBM(){};
    public GBM(ArrayList<Tree> trees, Loss loss, double first_round_pred, double eta){
        this.trees = trees;
        this.loss = loss;
        this.first_round_pred = first_round_pred;
        this.eta = eta;
    }

    public void fit(String file_training,
                    String file_validation,
                    ArrayList<String> categorical_features,
                    int early_stopping_rounds,
                    boolean maximize,
                    String eval_metric,
                    String loss,
                    double eta,
                    int num_boost_round,
                    int max_depth,
                    double scale_pos_weight,
                    double rowsample,
                    double colsample,
                    double min_child_weight,
                    int min_sample_split,
                    double lambda,
                    double gamma,
                    int num_thread){
        this.eta = eta;
        this.num_boost_round = num_boost_round;
        this.max_depth = max_depth;
        this.rowsample = rowsample;
        this.colsample = colsample;
        this.lambda = lambda;
        this.gamma = gamma;
        this.min_sample_split = min_sample_split;
        this.num_thread = num_thread;
        this.eval_metric = eval_metric;
        this.min_child_weight = min_child_weight;
        this.scale_pos_weight = scale_pos_weight;

        TrainData trainset = new TrainData(file_training,categorical_features);
        AttributeList attribute_list = new AttributeList(trainset);
        ClassList class_list = new ClassList(trainset);
        RowSampler row_sampler = new RowSampler(trainset.dataset_size,this.rowsample);
        ColumnSampler col_sampler = new ColumnSampler(trainset.feature_dim,this.colsample);
        trainset = null;

        if(loss.equals("logloss")){
            this.loss = new LogisticLoss();
            this.first_round_pred = 0.0;
        }else if(loss.equals("squareloss")){
            this.loss = new SquareLoss();
            this.first_round_pred = average(class_list.label);
        }
        class_list.initialize_pred(this.first_round_pred);

        class_list.update_grad_hess(this.loss,this.scale_pos_weight);

        //to evaluate on validation set and conduct early stopping
        boolean do_validation;
        ValidationData valset;
        double[] val_pred;
        if(file_validation.equals("")){
            do_validation = false;
            valset = null;
            val_pred = null;
        }else {
            do_validation = true;
            valset = new ValidationData(file_validation);
            val_pred = new double[valset.dataset_size];
            Arrays.fill(val_pred,this.first_round_pred);
        }

        double best_val_metric;
        int best_round=0;
        int become_worse_round=0;
        if(maximize){
            best_val_metric = -Double.MAX_VALUE;
        }else {
            best_val_metric = Double.MAX_VALUE;
        }

        //Start learning
        logger.info("TGBoost start training");
        for(int i=0;i<num_boost_round;i++){
            Tree tree = new Tree(min_sample_split,min_child_weight,max_depth,colsample,rowsample,
                                 lambda,gamma,num_thread,attribute_list.cat_features_cols);
            tree.fit(attribute_list,class_list,row_sampler,col_sampler);
            //when finish building this tree, update the class_list.pred, grad, hess
            class_list.update_pred(this.eta);
            class_list.update_grad_hess(this.loss, this.scale_pos_weight);


            //save this tree
            this.trees.add(tree);

            logger.log(Level.INFO,
                    String.format("current tree has %d nodes,including %d nan tree nodes",tree.nodes_cnt,tree.nan_nodes_cnt));

            //print training information
            if(eval_metric.equals("")){
                logger.log(Level.FINEST,String.format("TGBoost round %d",i));
            }else {
                double train_metric = calculate_metric(eval_metric,this.loss.transform(class_list.pred),class_list.label);


                if(!do_validation){
                    logger.log(Level.INFO,String.format("TGBoost round %d,train-%s:%.6f",i,eval_metric,train_metric));
                }else {
                    double[] cur_tree_pred = tree.predict(valset.origin_feature);
                    for(int n=0;n<val_pred.length;n++){
                        val_pred[n] += this.eta * cur_tree_pred[n];
                    }
                    double val_metric = calculate_metric(eval_metric,this.loss.transform(val_pred),valset.label);
                    logger.log(Level.INFO,String.format("TGBoost round %d,train-%s:%.6f,val-%s:%.6f",i,eval_metric,train_metric,eval_metric,val_metric));
                    //check whether to early stop
                    if(maximize){
                        if(val_metric>best_val_metric){
                            best_val_metric = val_metric;
                            best_round = i;
                            become_worse_round = 0;
                        }else {
                            become_worse_round += 1;
                        }
                        if(become_worse_round>early_stopping_rounds){
                            logger.log(Level.INFO,String.format("TGBoost training stop,best round is %d,best val-%s is %.6f",i,eval_metric,best_val_metric));
                            break;
                        }
                    }else{
                        if(val_metric<best_val_metric){
                            best_val_metric = val_metric;
                            best_round = i;
                            become_worse_round = 0;
                        }else {
                            become_worse_round += 1;
                        }
                        if(become_worse_round>early_stopping_rounds){
                            logger.log(Level.INFO,String.format("TGBoost training stop,best round is %d,best val-%s is %.6f",i,eval_metric,best_val_metric));
                            break;
                        }
                    }
                }
            }
        }
    }

    public double[] predict(float[][] features){
        logger.info("TGBoost start predicting...");
        double[] pred = new double[features.length];
        for(int i=0;i<pred.length;i++){
            pred[i] += first_round_pred;
        }
        for(Tree tree:this.trees){
            double[] cur_tree_pred = tree.predict(features);
            for(int i=0;i<pred.length;i++){
                pred[i] += this.eta * cur_tree_pred[i];
            }
        }
        return this.loss.transform(pred);
    }

    public void predict(String file_test,String file_output){
        TestData testdata = new TestData(file_test);
        double[] preds = this.predict(testdata.origin_feature);
        String[] strs = new String[preds.length];
        for(int i=0;i<strs.length;i++){
            strs[i] = String.valueOf(preds[i]);
        }
        String content = String.join("\n",strs);
        try{
            Files.write(Paths.get(file_output), content.getBytes());
        }catch (IOException e){
            e.printStackTrace();
        }

    }

    private double calculate_metric(String eval_metric,double[] pred,double[] label){
        if(eval_metric.equals("acc")){
            return Metric.accuracy(pred,label);
        }else if(eval_metric.equals("error")){
            return Metric.error(pred,label);
        }else if(eval_metric.equals("mse")){
            return Metric.mean_square_error(pred,label);
        }else if(eval_metric.equals("mae")){
            return Metric.mean_absolute_error(pred,label);
        }else if(eval_metric.equals("auc")){
            return Metric.auc(pred,label);
        }else {
            throw new NotImplementedException();
        }
    }


    private double average(double[] vals){
        double sum = 0.0;
        for(double v:vals){
            sum += v;
        }
        return sum/vals.length;
    }

    public double getFirst_round_pred(){
        return this.first_round_pred;
    }

    public double getEta(){
        return this.eta;
    }

    public Loss getLoss(){
        return this.loss;
    }

    public ArrayList<Tree> getTrees() {
        return this.trees;
    }
}
