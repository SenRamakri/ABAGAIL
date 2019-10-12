package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        String csvTime = "";
        String csvFitness = "";
        long starttime, sumOpt, sumTime;
        int loopCount = 1, i=0, j=0;
        int[] iterations = {50, 100, 500, 1000, 2000, 5000, 10000, 20000};
        csvTime += "AlgoTime";
        csvFitness += "AlgoOptimal";
        for(i=0; i<iterations.length;i++) {
            csvTime += ("," + Integer.toString(iterations[i]));
            csvFitness += ("," + Integer.toString(iterations[i]));  
        }
        csvFitness += "\n";
        csvTime += "\n";

        //////////////////////////////////////////////////////////////////////
        csvTime += "RHC";
        csvFitness += "RHC";
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(j=0; j<loopCount; j++) {
                starttime = System.currentTimeMillis();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[i]);
                fit.train();
                sumOpt += ef.value(rhc.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("RHC Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + (sumOpt / loopCount));
            csvTime += ("," + (sumTime / loopCount)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "SA";
        csvFitness += "SA";
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(j=0; j<loopCount; j++) {
                starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations[i]);
                fit.train();
                sumOpt += ef.value(sa.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("SA Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + (sumOpt / loopCount));
            csvTime += ("," + (sumTime / loopCount)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "GA";
        csvFitness += "GA";
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(j=0; j<loopCount; j++) {
                starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations[i]);
                fit.train();
                sumOpt += ef.value(ga.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("GA Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + (sumOpt / loopCount));
            csvTime += ("," + (sumTime / loopCount)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        /////////////////////////////////////////////////////////////////////////

        csvTime += "MIMIC";
        csvFitness += "MIMIC";
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(j=0; j<loopCount; j++) {
                starttime = System.currentTimeMillis();
                MIMIC mimic = new MIMIC(200, 100, pop);  
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iterations[i]);
                fit.train();
                sumOpt += ef.value(mimic.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("MIMIC Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + (sumOpt / loopCount));
            csvTime += ("," + (sumTime / loopCount)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ///////////////////////////////////////////////////////////////////////////
        
        System.out.println(csvFitness);
        System.out.println(csvTime);

        try {
            File file = new File("knapsack_opt.csv");
            file.createNewFile();
            BufferedWriter csvFitnessFile = new BufferedWriter(new FileWriter(file));
            csvFitnessFile.write(csvFitness);
            csvFitnessFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
        
        try {
            File file = new File("knapsack_time.csv");
            file.createNewFile();
            BufferedWriter csvTimeFile = new BufferedWriter(new FileWriter(file));
            csvTimeFile.write(csvTime);
            csvTimeFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}
