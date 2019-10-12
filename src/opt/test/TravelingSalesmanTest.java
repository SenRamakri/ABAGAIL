package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
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
                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
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
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);   
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
                // for mimic we use a sort encoding
                ef = new TravelingSalesmanSortEvaluationFunction(points);
                int[] ranges = new int[N];
                Arrays.fill(ranges, N);
                odd = new  DiscreteUniformDistribution(ranges);
                Distribution df = new DiscreteDependencyTree(.1, ranges); 
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                
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
            File file = new File("travelingsalesman_opt.csv");
            file.createNewFile();
            BufferedWriter csvFitnessFile = new BufferedWriter(new FileWriter(file));
            csvFitnessFile.write(csvFitness);
            csvFitnessFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
        
        try {
            File file = new File("travelingsalesman_time.csv");
            file.createNewFile();
            BufferedWriter csvTimeFile = new BufferedWriter(new FileWriter(file));
            csvTimeFile.write(csvTime);
            csvTimeFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}
