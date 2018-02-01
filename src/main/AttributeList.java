package main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class AttributeList {
    public int feature_dim;
    private float[][][] attribute_list;
    public int[][] missing_value_attribute_list;
    public int[][][] cutting_inds;
    public float[][] cutting_thresholds;
    public float[][] origin_feature;
    public ArrayList<Integer> cat_features_cols;

    public AttributeList(TrainData data){
        missing_value_attribute_list = data.missing_index;
        feature_dim = data.feature_dim;
        attribute_list = data.feature_value_index;
        origin_feature = data.origin_feature;
        cat_features_cols = data.cat_features_cols;
        sort_attribute_list();
        initialize_cutting_inds_thresholds();
        clean_up();
    }

    //pre-sort: for each feature,sort (value,index) by the value
    private void sort_attribute_list(){
        for(int i=0;i<feature_dim;i++){
            Arrays.sort(attribute_list[i], new Comparator<float[]>() {
                @Override
                public int compare(float[] a, float[] b) {
                    return Double.compare(a[0], b[0]);
                }
            });
        }
    }

    private void initialize_cutting_inds_thresholds(){
        cutting_inds = new int[feature_dim][][];
        cutting_thresholds = new float[feature_dim][];

        for(int i=0;i<feature_dim;i++){
            //for this feature, get its cutting index
            ArrayList<Integer> list = new ArrayList<>();
            int last_index = 0;
            for(int j=0;j<attribute_list[i].length;j++){
                if(attribute_list[i][j][0]==attribute_list[i][last_index][0]){
                    last_index = j;
                }else {
                    list.add(last_index);
                    last_index = j;
                }
            }
            //for this feature,store its cutting threshold
            cutting_thresholds[i] = new float[list.size()+1];
            for(int t=0;t<cutting_thresholds[i].length-1;t++){
                cutting_thresholds[i][t] = attribute_list[i][list.get(t)][0];
            }
            cutting_thresholds[i][list.size()] = attribute_list[i][list.get(list.size()-1)+1][0];

            //for this feature,store inds of each interval
            cutting_inds[i] = new int[list.size()+1][]; //list.size()+1 interval

            list.add(0,-1);
            list.add(attribute_list[i].length-1);
            for(int k=0;k<cutting_inds[i].length;k++){
                int start_ind = list.get(k)+1;
                int end_ind = list.get(k+1);
                cutting_inds[i][k] = new int[end_ind-start_ind+1];
                for(int m=0;m<cutting_inds[i][k].length;m++){
                    cutting_inds[i][k][m] = (int) attribute_list[i][start_ind+m][1];
                }
            }

        }
    }

    private void clean_up(){
        attribute_list = null;
    }


}
