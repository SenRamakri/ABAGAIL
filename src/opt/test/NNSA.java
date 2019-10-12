package opt.test;

/**
 * Created by Daniel Cai on 3/5/2017.
 */

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
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
public class NNSA {
    private static Instance[] instances = initializeInstances();
    private static int inputLayer = 14, hiddenLayer = 2, outputLayer = 1;
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 8239);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 8239, 14980);

    private static DataSet set = new DataSet(train_set);

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"SA"};
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



    public static void main(String[] args) {

        String final_result = "";


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }


        //oa[0] = new SimulatedAnnealing(1E11, .95, nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
        //oa[2] = new SimulatedAnnealing(1E11, .95, nnop[2]);

        int[] iterations = {10, 100, 500, 1000, 2500, 5000};

        double[] coolings = {0.15,0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95};

        for (int trainingIterations : iterations) {
            results = "";
            for (double cooling : coolings) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                oa[0] = new SimulatedAnnealing(1E11, cooling, nnop[0]);
                train(oa[0], networks[0], oaNames[0], trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[0].getOptimal();
                networks[0].setWeights(optimalInstance.getData());

                // Calculate Training Set Statistics //
                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[0].setInputValues(train_set[j].getData());
                    networks[0].run();

                    //predicted = Double.parseDouble(train_set[j].getLabel().toString());
                    //actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    actual = Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[0].getOutputValues().toString());

                    //System.out.println("actual is " + actual);
                    //System.out.println("predicted is " + predicted);

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTrain Results for SA:" + cooling + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                final_result = oaNames[0] + "," + trainingIterations + "," + cooling + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime);
                write_output_to_file("nn_sa.csv", final_result);

                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                for (int j = 0; j < test_set.length; j++) {
                    networks[0].setInputValues(test_set[j].getData());
                    networks[0].run();

                    actual = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[0].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for SA: " + cooling + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

                final_result = oaNames[0] + "," + trainingIterations + "," + cooling + "," + "testing accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime);
                write_output_to_file("nn_sa.csv", final_result);
            }
            System.out.println("results for iteration: " + trainingIterations + "---------------------------");
            System.out.println(results);
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        int trainingIterations = iteration;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }


            //System.out.println("training error :" + df.format(train_error)+", testing error: "+df.format(test_error));
        }
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
            System.out.println("\tEntry : " + i + " label :" + attributes[i][1][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
