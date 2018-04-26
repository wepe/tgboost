package main;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;

public class Loss {
    public double[] grad(double[] pred,double[] label){
        throw new NotImplementedException();
    }

    public double[] hess(double[] pred,double[] label){
        throw new NotImplementedException();
    }

    public double[] transform(double[] pred){
        throw new NotImplementedException();
    }
}


class SquareLoss extends Loss{
    public double[] transform(double[] pred){
        return pred;
    }

    public double[] grad(double[] pred,double[] label){
        double[] ret = new double[pred.length];
        for(int i=0;i<ret.length;i++){
            ret[i] = pred[i] - label[i];
        }
        return ret;
    }

    public double[] hess(double[] pred,double[] label){
        double[] ret = new double[pred.length];
        Arrays.fill(ret,1.0);
        return ret;
    }

}


class LogisticLoss extends Loss{
    private double clip(double val){
        if(val<0.00001){
            return 0.00001;
        }else if(val>0.99999){
            return 0.99999;
        }else {
            return val;
        }
    }

    public double[] transform(double[] pred){
        double[] ret = new double[pred.length];
        for(int i=0;i<ret.length;i++){
            ret[i] = clip(1.0 / (1.0+Math.exp(-pred[i])));
        }
        return ret;
    }

    public double[] grad(double[] pred,double[] label){
        double[] pred1 = transform(pred);
        double[] ret = new double[pred1.length];
        for(int i=0;i<ret.length;i++){
            ret[i] = pred1[i] - label[i];
        }
        return ret;
    }

    public double[] hess(double[] pred,double[] label){
        double[] pred1 = transform(pred);
        double[] ret = new double[pred.length];
        for(int i=0;i<ret.length;i++){
            ret[i] = pred1[i]*(1.0 - pred1[i]);
        }
        return ret;
    }
}