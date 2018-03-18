package main;

import java.util.ArrayList;

public class Main {
    public static void training(String[] args){
        String file_training = args[1];
        String file_validation = args[2];
        String file_model = args[3];
        int early_stopping_round = Integer.parseInt(args[4]);
        boolean maximize = args[5].equals("true");
        String eval_metric = args[6];
        String loss = args[7];
        double eta = Double.parseDouble(args[8]);
        int num_boost_round = Integer.parseInt(args[9]);
        int max_depth = Integer.parseInt(args[10]);
        double scale_pos_weight = Double.parseDouble(args[11]);
        double rowsample = Double.parseDouble(args[12]);
        double colample = Double.parseDouble(args[13]);
        double min_child_weight = Double.parseDouble(args[14]);
        int min_sample_split = Integer.parseInt(args[15]);
        double lambda = Double.parseDouble(args[16]);
        double gamma = Double.parseDouble(args[17]);
        int num_thread = Integer.parseInt(args[18]);

        String[] cat_features = args[19].split(",");
        ArrayList<String> categorical_features = new ArrayList<>();
        for(String cat_feature:cat_features){
            categorical_features.add(cat_feature);
        }

        GBM tgb = new GBM();
        tgb.fit(file_training,
                file_validation,
                categorical_features,
                early_stopping_round,
                maximize,
                eval_metric,
                loss,
                eta,
                num_boost_round,
                max_depth,
                scale_pos_weight,
                rowsample,
                colample,
                min_child_weight,
                min_sample_split,
                lambda,
                gamma,
                num_thread);

        ModelSerializer.save_model(tgb,file_model);
    }

    public static void testing(String[] args){
        String file_model = args[1];
        String file_testing = args[2];
        String file_output = args[3];

        GBM tgb = ModelSerializer.load_model(file_model);
        tgb.predict(file_testing, file_output);
    }

    public static void main(String[] args){
        if(args[0].equals("training")){
            training(args);
        }else if(args[0].equals("testing")){
            testing(args);
        }
    }
}
