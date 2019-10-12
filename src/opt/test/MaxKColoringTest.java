package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
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
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoringTest {
    /** The n value */
    private static final int N = 50; // number of vertices
    private static final int L =4; // L adjacent nodes per vertex
    private static final int K = 8; // K possible colors
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random(N*L);
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;	
            vertex.setAdjMatrixSize(L);
            for(int j = 0; j <L; j++ ){
            	 vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
            }
        }
        /*for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }*/
        // for rhc, sa, and ga we use a permutation based encoding
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        Distribution df = new DiscreteDependencyTree(.1); 
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
            csvFitness += ("," + ef.foundMax());
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
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations[i]);
                fit.train();
                sumOpt += ef.value(sa.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("SA Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + ef.foundMax());
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
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 60, gap);      
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations[i]);
                fit.train();
                sumOpt += ef.value(ga.getOptimal());
                sumTime += System.currentTimeMillis() - starttime;
            }
            System.out.println("GA Optimal : " + sumOpt / loopCount);
            System.out.println("Time : " + sumTime / loopCount);
            csvFitness += ("," + ef.foundMax());
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
            csvFitness += ("," + ef.foundMax());
            csvTime += ("," + (sumTime / loopCount)); 
        }
        csvFitness += "\n";
        csvTime += "\n";
        ///////////////////////////////////////////////////////////////////////////
        
        System.out.println(csvFitness);
        System.out.println(csvTime);

        try {
            File file = new File("maxkcolor_opt.csv");
            file.createNewFile();
            BufferedWriter csvFitnessFile = new BufferedWriter(new FileWriter(file));
            csvFitnessFile.write(csvFitness);
            csvFitnessFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
        
        try {
            File file = new File("maxkcolor_time.csv");
            file.createNewFile();
            BufferedWriter csvTimeFile = new BufferedWriter(new FileWriter(file));
            csvTimeFile.write(csvTime);
            csvTimeFile.close();
        } catch (IOException ioe) {
	        ioe.printStackTrace();
	    }
    }
}
