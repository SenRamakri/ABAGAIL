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
import java.io.File;
import java.io.IOException;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    private static final int N = 80;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        String csvTime = "";
        String csvFitness = "";
        long starttime, sumOpt, sumTime;
        int loopCount = 5, i=0, j=0;
        int[] iterations = {50, 100, 500, 1000, 2000, 5000, 10000};
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
            starttime = System.currentTimeMillis();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations[i]);
            fit.train();
            //System.out.println("RHC Optimal single: " + ef.value(rhc.getOptimal()));
            sumOpt += ef.value(rhc.getOptimal());
            sumTime += System.currentTimeMillis() - starttime;
            System.out.println("RHC Optimal : " + (float)sumOpt);
            System.out.println("Time : " + sumTime);
            csvFitness += ("," + ((float)sumOpt));
            csvTime += ("," + (sumTime)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "SA";
        csvFitness += "SA";
        double coolings[] = { 0.2, 0.4, 0.6, 0.8, 0.9 };
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(j=0; j<coolings.length; j++) {
                starttime = System.currentTimeMillis();
                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, coolings[j], hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations[i]);
                fit.train();
                //System.out.println("SA Optimal single: " + ef.value(sa.getOptimal()));
                csvTime += ("," + ((float)coolings[j]));
                csvFitness += ("," + ((float)coolings[j]));
                sumOpt += ef.value(sa.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
                System.out.println("SA Optimal : " + (float)sumOpt);
                System.out.println("Time : " + sumTime);
                csvFitness += ("," + ((float)sumOpt));
                csvTime += ("," + (sumTime)); 
            }
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "GA";
        csvFitness += "GA";
        int[] mate = {25, 50, 100, 150, 200};
        int[] mute = {1, 2, 3, 4, 5};
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(int mates=0; mates<mate.length; mates++) {
                for(int mutes=0; mutes<mute.length; mutes++) {
                    starttime = System.currentTimeMillis();
                    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, mate[mates], mute[mutes], gap);   
                    FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations[i]);
                    fit.train();
                    sumOpt += ef.value(ga.getOptimal());
                    sumTime += System.currentTimeMillis() - starttime;
                    csvTime += ("," + mate[mates] + "," + mute[mutes]);
                    csvFitness += ("," + mate[mates] + "," + mute[mutes]);
                    System.out.println("GA Optimal : " + (float)sumOpt);
                    System.out.println("Time : " + sumTime);
                    csvFitness += ("," + ((float)sumOpt));
                    csvTime += ("," + (sumTime));
                } 
            }
        }
        csvFitness += "\n";
        csvTime += "\n";
        
        /////////////////////////////////////////////////////////////////////////

        int[] samples = {100, 150, 200, 250, 300};
        int[] tokeep = {20, 30, 40, 50, 60};
        csvTime += "MIMIC";
        csvFitness += "MIMIC";
        for(i=0; i<iterations.length;i++) {
            sumOpt = 0;
            sumTime = 0;
            for(int samps=0; samps<samples.length; samps++) {
                for(int tokeeps=0; tokeeps<tokeep.length; tokeeps++) {
                    starttime = System.currentTimeMillis();
                    MIMIC mimic = new MIMIC(samples[samps], tokeep[tokeeps], pop);
                    //MIMIC mimic = new MIMIC(100, 50, pop);
                    FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iterations[i]);
                    fit.train();
                    sumOpt += ef.value(mimic.getOptimal());
                    sumTime += System.currentTimeMillis() - starttime;
                    csvTime += ("," + samples[samps] + "," + tokeep[tokeeps]);
                    csvFitness += ("," + samples[samps] + "," + tokeep[tokeeps]);
                    System.out.println("MIMIC Optimal : " + (float)sumOpt);
                    System.out.println("Time : " + sumTime);
                    csvFitness += ("," + ((float)sumOpt));
                    csvTime += ("," + (sumTime));
                }
            }
        }
        csvFitness += "\n";
        csvTime += "\n";
        ///////////////////////////////////////////////////////////////////////////
        
        System.out.println(csvFitness);
        System.out.println(csvTime);

        try {
            File file = new File("countones_opt.csv");
            file.createNewFile();
            BufferedWriter csvFitnessFile = new BufferedWriter(new FileWriter(file));
            csvFitnessFile.write(csvFitness);
            csvFitnessFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
        
        try {
            File file = new File("countones_time.csv");
            file.createNewFile();
            BufferedWriter csvTimeFile = new BufferedWriter(new FileWriter(file));
            csvTimeFile.write(csvTime);
            csvTimeFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}