package opt.test;

import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
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

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
		EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        String csvTime = "";
        String csvFitness = "";
        long starttime, sumOpt, sumTime;
        int loopCount = 1, i=0, j=0;
        //int[] iterations = {100, 500, 1000, 2000, 5000, 10000, 50000};
        int[] iterations = {100, 500, 1000};
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
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);     
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
                MIMIC mimic = new MIMIC(200, 20, pop);    
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
            BufferedWriter csvFitnessFile = new BufferedWriter(new FileWriter("fourpeaks_opt.csv", true));
            csvFitnessFile.write(csvFitness);
            csvFitnessFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
        
        try {
            BufferedWriter csvTimeFile = new BufferedWriter(new FileWriter("fourpeaks_time.csv", true));
            csvTimeFile.write(csvTime);
            csvTimeFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}
