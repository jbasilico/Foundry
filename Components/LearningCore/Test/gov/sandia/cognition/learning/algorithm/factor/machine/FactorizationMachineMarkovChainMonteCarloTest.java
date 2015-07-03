/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.learning.algorithm.factor.machine.FactorizationMachineMarkovChainMonteCarlo;
import gov.sandia.cognition.learning.algorithm.factor.machine.FactorizationMachineAlternatingLeastSquares;
import gov.sandia.cognition.learning.algorithm.factor.machine.FactorizationMachine;
import gov.sandia.cognition.algorithm.IterativeAlgorithm;
import gov.sandia.cognition.algorithm.event.AbstractIterativeAlgorithmListener;
import gov.sandia.cognition.learning.data.DefaultInputOutputPair;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.learning.performance.MeanSquaredErrorEvaluator;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.util.NamedValue;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for class {@link FactorizationMachineMarkovChainMonteCarlo}.
 * @author  Justin Basilico
 * @since   3.4.0
 */
public class FactorizationMachineMarkovChainMonteCarloTest
    extends Object
{
    public static final NumberFormat NUMBER_FORMAT = new DecimalFormat("0.0000");
    protected Random random = new Random(47474747);
    
    /**
     * Creates a new test.
     */
    public FactorizationMachineMarkovChainMonteCarloTest()
    {
        super();
    }
    
    /**
     * Test of constructors, of class FactorizationMachineAlternatingLeastSquares.
     */
    @Test
    public void testConstructors()
    {
        FactorizationMachineAlternatingLeastSquares instance =
            new FactorizationMachineAlternatingLeastSquares();
        fail("The test case is a prototype.");
    }

    /**
     * Test of learn method, of class FactorizationMachineAlternatingLeastSquares.
     */
    @Test
    public void testLearn()
    {
        System.out.println("learn");
        boolean useBias = true;
        boolean useWeights = true;
        boolean useFactors = true;
        int n = 1000;
        int d = 20;
        int k = 10;
        FactorizationMachine actual = new FactorizationMachine(d, k);
        actual.setBias(this.random.nextGaussian() * 10.0 * (useBias ? 1.0 : 0.0));
        actual.setWeights(VectorFactory.getDenseDefault().createUniformRandom(d,
            -1.0, 1.0, this.random).scale(useWeights ? 1.0 : 0.0));
        actual.setFactors(MatrixFactory.getDenseDefault().createUniformRandom(k,
            d, -1.0, 1.0, this.random).scale(useFactors ? 1.0 : 0.0));
        
        int trainSize = n;
        int testSize = n;
        int totalSize = trainSize + testSize;
        List<InputOutputPair<Vector, Double>> trainData = new ArrayList<InputOutputPair<Vector, Double>>();
        final List<InputOutputPair<Vector, Double>> testData = new ArrayList<InputOutputPair<Vector, Double>>();
        
        for (int i = 0; i < totalSize; i++)
        {
            Vector input = VectorFactory.getDenseDefault().createUniformRandom(
                d, -10.0, 10.0, this.random);
            final DefaultInputOutputPair<Vector, Double> example =
                DefaultInputOutputPair.create(input, actual.evaluateAsDouble(input));
            if (i < trainSize)
            {
                trainData.add(example);
            }
            else
            {
                testData.add(example);
            }
        }
        
        FactorizationMachineMarkovChainMonteCarlo instance =
            new FactorizationMachineMarkovChainMonteCarlo();
        instance.setFactorCount(useFactors ? k : 0);
        instance.setSeedScale(0.2);
        instance.setMaxIterations(1000);
        instance.setWeightsEnabled(useWeights);
        instance.setBiasEnabled(useBias);
        instance.setRandom(random);
//        instance.addIterativeAlgorithmListener(new IterationMeasurablePerformanceReporter());
// TODO: Part of this may be good as a general class (printing validation metrics).
        instance.addIterativeAlgorithmListener(new AbstractIterativeAlgorithmListener()
        {

            @Override
            public void stepEnded(IterativeAlgorithm algorithm)
            {
                final FactorizationMachineMarkovChainMonteCarlo a = (FactorizationMachineMarkovChainMonteCarlo) algorithm;
                MeanSquaredErrorEvaluator<Vector> performance =
                    new MeanSquaredErrorEvaluator<Vector>();
                System.out.println("Iteration " + a.getIteration() 
                    + " RMSE: Train: " + NUMBER_FORMAT.format(Math.sqrt(performance.evaluatePerformance(a.getResult(), a.getData())))
                    + " Validation: " + NUMBER_FORMAT.format(Math.sqrt(performance.evaluatePerformance(a.getResult(), testData)))
                    + " Alpha: " + a.getAlpha()
                    + " Sample RMSE: Train: " + NUMBER_FORMAT.format(Math.sqrt(performance.evaluatePerformance(a.getSample(), a.getData())))
                    + " Validation: " + NUMBER_FORMAT.format(Math.sqrt(performance.evaluatePerformance(a.getSample(), testData)))
                    + " Lambdas: " + NUMBER_FORMAT.format(a.biasLambda) + " " + NUMBER_FORMAT.format(a.weightsLambda) + " " + a.factorsLambda.toString(NUMBER_FORMAT, ", ")
);
// TODO: Expose more of the details from the MCMC run.
//                    + " Objective: " + NUMBER_FORMAT.format(a.getObjective())
//                    + " Change: " + NUMBER_FORMAT.format(a.getTotalChange())
//                    + " Error: " + NUMBER_FORMAT.format(Math.sqrt(a.getTotalError() / a.getData().size()))
//                    + " Regularization: " + NUMBER_FORMAT.format(a.getRegularizationPenalty()));
            }
            
        });
        
// TODO: Figure out why this doesn't work with real factors.
        FactorizationMachine result = instance.learn(trainData);
        System.out.println(actual.getBias());
        System.out.println(actual.getWeights());
        System.out.println(actual.getFactors());
        System.out.println(result.getBias());
        System.out.println(result.getWeights());
        System.out.println(result.getFactors());
        
        MeanSquaredErrorEvaluator<Vector> performance = new MeanSquaredErrorEvaluator<>();
        System.out.println("RMSE: " + Math.sqrt(performance.evaluatePerformance(result, testData)));
        assertTrue(Math.sqrt(performance.evaluatePerformance(result, testData)) < 0.05);
        fail("The test case is a prototype.");
    }
    
    /**
     * Test of getPerformance method, of class FactorizationMachineMarkovChainMonteCarlo.
     */
    @Test
    public void testGetPerformance()
    {
        System.out.println("getPerformance");
        FactorizationMachineMarkovChainMonteCarlo instance =
            new FactorizationMachineMarkovChainMonteCarlo();
        NamedValue<?> expResult = null;
        NamedValue<?> result = instance.getPerformance();
        assertEquals(expResult, result);
        fail("The test case is a prototype.");
    }
    
}