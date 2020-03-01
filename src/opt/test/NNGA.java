package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import shared.FixedIterationTrainer;
import shared.tester.AccuracyTestMetric;
import shared.tester.CrossValidationTestMetric;
import shared.tester.NeuralNetworkTester;
import shared.tester.TestMetric;
import shared.tester.Tester;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class NNGA {
    private static Instance[] instances = initializeInstances();
    private static int inputLayer = 14, hiddenLayer = 10, outputLayer = 1;
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 11984);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 11984, 14980);
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static FeedForwardNetwork networks;
    private static NeuralNetworkOptimizationProblem nnop;

    private static OptimizationAlgorithm oa;
    private static String oaNames = "GA";
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void write_output_to_file(String file_name, String results) {
        try {
            File file = new File(file_name);
            if (!file.exists()) {
                file.createNewFile();
            }
            PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(file, true)));
            synchronized (pwtr) {
                pwtr.println(results);
                pwtr.close();
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void init_output_file() {
        try {
            File file = new File("nn_ga.csv");
            if (file.exists()) {
                file.delete();
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {

        String final_result = "";
        init_output_file();
        //make CV folds
        int foldsize = 5;
        int foldsetsize = train_set.length / foldsize;
        int cvsetsize = train_set.length - foldsetsize;

        int[] mate = {5, 10, 25, 50, 100};
        
        for(int f=0 ; f<5; f++ ) {
            int foldstart = f * foldsetsize;
            int foldend = (f+1) * foldsetsize;
            Instance[] cv_train_set_1 = Arrays.copyOfRange(instances, 0, foldstart);
            Instance[] cv_train_set_2 = Arrays.copyOfRange(instances, foldend, train_set.length - 1);
            Instance[] cv_train_set = new Instance[cv_train_set_1.length + cv_train_set_2.length];
            System.arraycopy(cv_train_set_1, 0, cv_train_set, 0, cv_train_set_1.length);
            System.arraycopy(cv_train_set_2, 0, cv_train_set, cv_train_set_1.length, cv_train_set_2.length);
            DataSet set = new DataSet(cv_train_set);
            System.out.println("Start: " + foldstart + " end: " + foldend + " foldsetsize: " + foldsetsize + " cvtrainsetsize: " + cv_train_set.length);
        
            RELU relu = new RELU();
            networks = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer}, relu);
            nnop = new NeuralNetworkOptimizationProblem(set, networks, measure);

            for(int q=0; q<5; q++) {
                oa = new StandardGeneticAlgorithm(200, mate[q], 1, nnop);
                results = "";
                
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                // 7) Instantiate a trainer.  The FixtIterationTrainer takes another trainer (in this case,
                //    an OptimizationAlgorithm) and executes it a specified number of times.
                FixedIterationTrainer fit = new FixedIterationTrainer(oa, 5000);
                
                // 8) Run the trainer.  This may take a little while to run, depending on the OptimizationAlgorithm,
                //    size of the data, and number of iterations.
                fit.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa.getOptimal();
                networks.setWeights(optimalInstance.getData());
                //System.out.println("optimalInstance.getData(): " + optimalInstance.getData());

                // Calculate Training Set Statistics //
                double predicted, actual;
                
                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                for (int j = 0; j < test_set.length; j++) {
                    networks.setInputValues(test_set[j].getData());
                    networks.run();

                    actual = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks.getOutputValues().toString());

                    System.out.println("vals: " + actual + ":" + predicted + ":" + Math.round(predicted));

                    double trash = (Math.round(predicted) == actual) ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for " + oaNames + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining/Optimization time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                final_result += (oaNames + "," + "mate" + "," + mate[q] + "," + "testing accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                                + "," + "training/optimization time" + "," + df.format(trainingTime) + "," + "testing time" +
                                "," + df.format(testingTime)) + "\n";
                System.out.println(results);
            }
        }
        write_output_to_file("nn_ga.csv", final_result);
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[14980][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("./eeg_dataset.csv")));
            String header = br.readLine();
            System.out.println("Header : " + header);
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[14]; // 14 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 14; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        System.out.println("Instances len : " + instances.length);

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // already classified as 1,0, just assign it
            //System.out.println("\tEntry : " + i + " label :" + attributes[i][1][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
