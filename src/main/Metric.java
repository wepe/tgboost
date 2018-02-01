package main;

import java.util.Arrays;
import java.util.Comparator;

public class Metric {
    public static double accuracy(double[] pred,double[] label){
        double hit=0.0;
        for(int i=0;i<pred.length;i++){
            if((label[i]==0 && pred[i]<0.5) || (label[i]==1 && pred[i]>0.5)){
                hit++;
            }
        }
        return hit/pred.length;
    }

    public static double error(double[] pred,double[] label){
        return 1.0 - accuracy(pred,label);
    }

    public static double mean_square_error(double[] pred,double[] label){
        double sum = 0.0;
        for(int i=0;i<pred.length;i++){
            sum += Math.pow(pred[i] - label[i],2.0);
        }
        return sum/pred.length;
    }

    public static double mean_absolute_error(double[] pred,double[] label){
        double sum = 0.0;
        for(int i=0;i<pred.length;i++){
            sum += Math.abs(pred[i]-label[i]);
        }
        return sum/pred.length;
    }

    public static double auc(double[] pred,double[] label){
        double n_pos = 0;
        for(double v:label) n_pos+=v;
        double n_neg = pred.length - n_pos;

        double[][] label_pred = new double[pred.length][2];
        for(int i=0;i<pred.length;i++){
            label_pred[i][0] = label[i];
            label_pred[i][1] = pred[i];
        }

        Arrays.sort(label_pred, new Comparator<double[]>() {
            @Override
            public int compare(double[] a, double[] b) {
                return Double.compare(a[1],b[1]);
            }
        });

        double accumulated_neg = 0;
        double satisfied_pair = 0;
        for(int i=0;i<label_pred.length;i++){
            if(label_pred[i][0] == 1){
                satisfied_pair += accumulated_neg;
            }else {
                accumulated_neg += 1;
            }
        }
        return satisfied_pair / n_neg / n_pos;

    }


}
