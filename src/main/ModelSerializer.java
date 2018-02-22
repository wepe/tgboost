package main;

/*
Serialize the GBM model into txt file, unserialize the txt file into GBM model.

the content format in the txt file:
    first_round_prediction
    tree[tree_index]:
    internal_node_index:[feature_name,feature_type,split value || split values],missing_go_to=0|1|2
    leaf_node_index:leaf=leaf_score

for example:
    0.5000
    tree[0]:
    1:[7,num,30.6000],missing_go_to=0
    2:[9,cat,1,3,5],missing_go_to=1
    4:leaf=0.3333

    tree[1]:
    1:[4,num,10.9000],missing_go_to=2
    2:leaf=0.9900
 */

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class ModelSerializer {
    private static String serializeLeafNode(TreeNode node){
        StringBuilder sb = new StringBuilder();
        sb.append(node.index);
        sb.append(":leaf=");
        sb.append(String.format("%.6f", node.leaf_score));
        return sb.toString();
    }

    private static String serializeInternalNode(TreeNode node){
        StringBuilder sb = new StringBuilder();
        sb.append(node.index);
        sb.append(":[");
        sb.append(node.split_feature+",");

        if(node.split_left_child_catvalue==null){
            sb.append("num,");
            sb.append(String.format("%.6f",node.split_threshold));
            sb.append("],");
        }else {
            sb.append("cat");
            for(double catvalue:node.split_left_child_catvalue){
                sb.append(","+catvalue);
            }
            sb.append("],");
        }

        if(node.nan_go_to==0){
            sb.append("missing_go_to=0");
        }else if(node.nan_go_to==1){
            sb.append("missing_go_to=1");
        }else if(node.nan_go_to==2){
            sb.append("missing_go_to=2");
        }else{
            if(node.left_child.num_sample>node.right_child.num_sample){
                sb.append("missing_go_to=1");
            }else {
                sb.append("missing_go_to=2");
            }
        }
        return sb.toString();


    }

    //Serialize the GBM model into txt file
    public static void save_model(GBM gbm,String path){
        double first_round_predict = gbm.getFirst_round_pred();
        double eta = gbm.getEta();
        Loss loss = gbm.getLoss();
        ArrayList<Tree> trees = gbm.getTrees();

        StringBuilder sb = new StringBuilder();
        sb.append("first_round_predict="+first_round_predict+"\n");
        sb.append("eta="+eta+"\n");
        if(loss instanceof LogisticLoss){
            sb.append("logloss"+"\n");
        }else {
            sb.append("squareloss"+"\n");
        }

        for(int i=1;i<=trees.size();i++){
            sb.append("tree["+i+"]:\n");

            Tree tree = trees.get(i-1);
            TreeNode root = tree.getRoot();
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            while(!queue.isEmpty()){
                int cur_level_num = queue.size();

                while (cur_level_num!=0){
                    cur_level_num--;
                    TreeNode node = queue.poll();
                    if(node.is_leaf){
                        sb.append(serializeLeafNode(node)+"\n");
                    }else {
                        sb.append(serializeInternalNode(node)+"\n");
                        queue.offer(node.left_child);
                        if(node.nan_child!=null){
                            queue.offer(node.nan_child);
                        }
                        queue.offer(node.right_child);
                    }
                }

            }
        }
        sb.append("tree[end]");

        try{
            Files.write(Paths.get(path), sb.toString().getBytes());
        }catch (IOException e){
            e.printStackTrace();
        }

    }

    //unserialize the txt file into GBM model.
    public static GBM load_model(String path){
        try{
            BufferedReader br = new BufferedReader(new FileReader(path));
            double first_round_predict = Double.parseDouble(br.readLine().split("=")[1]);
            double eta = Double.parseDouble(br.readLine().split("=")[1]);
            Loss loss = null;
            if(br.readLine().equals("logloss")){
                loss = new LogisticLoss();
            }else {
                loss = new SquareLoss();
            }

            ArrayList<Tree> trees = new ArrayList<>();
            String line;
            HashMap<Integer,TreeNode> map = new HashMap<>();
            while ((line=br.readLine())!=null){
                if(line.startsWith("tree")){
                    //store this tree,clear map
                    if(!map.isEmpty()){
                        Queue<TreeNode> queue = new LinkedList<>();
                        TreeNode root = map.get(1);
                        queue.offer(root);
                        while (!queue.isEmpty()){
                            int cur_level_num = queue.size();
                            while(cur_level_num!=0){
                                cur_level_num--;
                                TreeNode node = queue.poll();
                                if(!node.is_leaf){
                                    node.left_child = map.get(3*node.index-1);
                                    node.right_child = map.get(3*node.index+1);
                                    queue.offer(node.left_child);
                                    queue.offer(node.right_child);
                                    if(map.containsKey(3*node.index)){
                                        node.nan_child = map.get(3*node.index);
                                        queue.offer(node.nan_child);
                                    }
                                }
                            }
                        }

                        trees.add(new Tree(root));
                        map.clear();
                    }
                }else {
                    //store this node into map
                    int index = Integer.parseInt(line.split(":")[0]);
                    if(line.split(":")[1].startsWith("leaf")){
                        double leaf_score = Double.parseDouble(line.split(":")[1].split("=")[1]);
                        TreeNode node = new TreeNode(index,leaf_score);
                        map.put(index,node);
                    }else {
                        double nan_go_to = Double.parseDouble(line.split("=")[1]);
                        String split_info = line.split(":")[1].split("]")[0];
                        split_info = split_info.substring(1);
                        String[] strs = split_info.split(",");
                        int split_feature = Integer.parseInt(strs[0]);
                        if(strs[1].equals("num")){
                            double split_threshold = Double.parseDouble(strs[2]);
                            TreeNode node = new TreeNode(index,split_feature,split_threshold,nan_go_to);
                            map.put(index,node);
                        }else {
                            ArrayList<Double> split_left_child_catvalue = new ArrayList<>();
                            for(int i=2;i<strs.length;i++){
                                split_left_child_catvalue.add(Double.parseDouble(strs[i]));
                            }
                            TreeNode node = new TreeNode(index,split_feature,split_left_child_catvalue,nan_go_to);
                            map.put(index,node);
                        }
                    }
                }
            }
            return new GBM(trees,loss,first_round_predict,eta);
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }
}
