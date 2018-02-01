package main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class RowSampler{
    public ArrayList<Double> row_mask = new ArrayList<>();

    public RowSampler(int n, double sampling_rate){
        for(int i=0;i<n;i++){
            this.row_mask.add(Math.random()<=sampling_rate? 1.0 : 0.0);
        }
    }

    public void shuffle(){
        Collections.shuffle(this.row_mask);
    }
}


class ColumnSampler{
    private ArrayList<Integer> cols = new ArrayList<>();
    public List<Integer> col_selected;
    private int n_selected;

    public ColumnSampler(int n, double sampling_rate){
        for(int i=0;i<n;i++){
            cols.add(i);
        }
        n_selected = (int) (n * sampling_rate);
        col_selected = cols.subList(0,n_selected);
    }

    public void shuffle(){
        Collections.shuffle(cols);
        col_selected = cols.subList(0,n_selected);
    }
}

public class Sampling {
    public static void main(String[] args) {
        //test case
        RowSampler rs = new RowSampler(1000000, 0.8);
        System.out.println(rs.row_mask.subList(0,20));
        rs.shuffle();
        System.out.println(rs.row_mask.subList(0,20));
        int sum = 0;
        for(double v:rs.row_mask){
            sum += v;
        }
        System.out.println(sum);

        ColumnSampler cs = new ColumnSampler(1000, 0.6);
        System.out.println(cs.col_selected.subList(0,20));
        cs.shuffle();
        System.out.println(cs.col_selected.subList(0,20));
        System.out.println(cs.col_selected.size());
    }
}