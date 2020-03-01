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
        int[] iterations = {50, 100, 500, 1000, 2000, 5000};
        //int[] iterations = {50, 100, 500};
                
        //////////////////////////////////////////////////////////////////////
        csvTime += "RHC\n";
        csvFitness += "RHC\n";
        csvTime += "Iter,Timeval\n";
        csvFitness += "Iter,Optval\n";
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
            csvFitness += (iterations[i] + "," + ((float)sumOpt)) + "\n";
            csvTime += (iterations[i] + "," + (sumTime)) + "\n"; 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "SA\n";
        csvFitness += "SA\n";
        csvTime += "Iter,Coolings,Timeval\n";
        csvFitness += "Iter,Coolings,Optval\n";
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
                csvTime += (iterations[i] + "," + ((float)coolings[j]));
                csvFitness += (iterations[i] + "," + ((float)coolings[j]));
                sumOpt += ef.value(sa.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
                System.out.println("SA Optimal : " + (float)sumOpt);
                System.out.println("Time : " + sumTime);
                csvFitness += ("," + ((float)sumOpt)) + "\n";
                csvTime += ("," + (sumTime)) + "\n"; 
            }
        }
        csvFitness += "\n";
        csvTime += "\n";
        ////////////////////////////////////////////////////////////////////////

        csvTime += "GA\n";
        csvFitness += "GA\n";
        csvTime += "Iter,Mate,Timeval\n";
        csvFitness += "Iter,Mate,Optval\n";
        int[] mate = {25, 50, 100, 150, 200};
        int[] mute = {1};
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
                    csvTime += (iterations[i] + "," + mate[mates]);
                    csvFitness += (iterations[i] + "," + mate[mates]);
                    System.out.println("GA Optimal : " + (float)sumOpt);
                    System.out.println("Time : " + sumTime);
                    csvFitness += ("," + ((float)sumOpt)) + "\n";
                    csvTime += ("," + (sumTime)) + "\n";
                } 
            }
        }
        csvFitness += "\n";
        csvTime += "\n";
        
        /////////////////////////////////////////////////////////////////////////

        int[] samples = {60, 100, 120, 150, 200};
        int[] tokeep = {50};
        csvTime += "MIMIC\n";
        csvFitness += "MIMIC\n";
        csvTime += "Iter,Samples,Timeval\n";
        csvFitness += "Iter,Samples,Optval\n";
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
                    csvTime += (iterations[i] + "," + samples[samps]);
                    csvFitness += (iterations[i] + "," + samples[samps]);
                    System.out.println("MIMIC Optimal : " + (float)sumOpt);
                    System.out.println("Time : " + sumTime);
                    csvFitness += ("," + ((float)sumOpt)) + "\n";
                    csvTime += ("," + (sumTime)) + "\n";
                    System.out.println("Args : " + samples[samps] + ":" + tokeep[tokeeps] + ":" + iterations[i]);
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

    public static void write_data_file(String filename, String content) {
        try {
            File file = new File(filename);
            file.createNewFile();
            BufferedWriter bufwriter = new BufferedWriter(new FileWriter(file));
            bufwriter.write(content);
            bufwriter.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}