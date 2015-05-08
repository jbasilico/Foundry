/*
 * File:            FactorizationMachinePairwiseStochasticGradientTest.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */


package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.IterativeAlgorithm;
import gov.sandia.cognition.algorithm.event.AbstractIterativeAlgorithmListener;
import gov.sandia.cognition.collection.DefaultMultiCollection;
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
 * Unit tests for class {@link FactorizationMachinePairwiseStochasticGradient}.
 * @author  Justin Basilico
 * @since   3.4.1
 */
public class FactorizationMachinePairwiseStochasticGradientTest
    extends Object
{
    protected final NumberFormat NUMBER_FORMAT = new DecimalFormat("0.0000");
    protected Random random = new Random(846463);
    
    /**
     * Creates a new test.
     */
    public FactorizationMachinePairwiseStochasticGradientTest()
    {
        super();
    }

    /**
     * Test of constructors of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testConstructors()
    {

        int factorCount = FactorizationMachinePairwiseStochasticGradient.DEFAULT_FACTOR_COUNT;
        double learningRate = FactorizationMachinePairwiseStochasticGradient.DEFAULT_LEARNING_RATE;
        double biasRegularization = FactorizationMachinePairwiseStochasticGradient.DEFAULT_BIAS_REGULARIZATION;
        double weightRegularization = FactorizationMachinePairwiseStochasticGradient.DEFAULT_WEIGHT_REGULARIZATION;
        double factorRegularization = FactorizationMachinePairwiseStochasticGradient.DEFAULT_FACTOR_REGULARIZATION;
        double seedScale = FactorizationMachinePairwiseStochasticGradient.DEFAULT_SEED_SCALE;
        int maxIterations = FactorizationMachinePairwiseStochasticGradient.DEFAULT_MAX_ITERATIONS;
        FactorizationMachinePairwiseStochasticGradient instance =
            new FactorizationMachinePairwiseStochasticGradient();
        assertEquals(factorCount, instance.getFactorCount());
        assertEquals(learningRate, instance.getLearningRate(), 0.0);
        assertEquals(biasRegularization, instance.getBiasRegularization(), 0.0);
        assertEquals(weightRegularization, instance.getWeightRegularization(), 0.0);
        assertEquals(factorRegularization, instance.getFactorRegularization(), 0.0);
        assertEquals(seedScale, instance.getSeedScale(), 0.0);
        assertEquals(maxIterations, instance.getMaxIterations());
        assertNotNull(instance.getRandom());
        assertSame(instance.getRandom(), instance.getRandom());
        
        factorCount = 22;
        learningRate = 0.12321;
        biasRegularization = 3.33;
        weightRegularization = 44.44;
        factorRegularization = 555.55;
        seedScale = 0.6;
        maxIterations = 777;
        Random random = new Random();
        instance = new FactorizationMachinePairwiseStochasticGradient(factorCount,
            learningRate, biasRegularization, weightRegularization, factorRegularization,
            seedScale, maxIterations, random);
        assertEquals(factorCount, instance.getFactorCount());
        assertEquals(biasRegularization, instance.getBiasRegularization(), 0.0);
        assertEquals(weightRegularization, instance.getWeightRegularization(), 0.0);
        assertEquals(factorRegularization, instance.getFactorRegularization(), 0.0);
        assertEquals(seedScale, instance.getSeedScale(), 0.0);
        assertEquals(maxIterations, instance.getMaxIterations());
        assertSame(random, instance.getRandom());
        
        // No negative factor counts.
        boolean exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(-1, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        
        // No zero learning rate.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, 0,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        // No negative bias regularization.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, learningRate,
                -1.0, weightRegularization, factorRegularization,
                seedScale, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        // No negative weight regularization.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, learningRate,
                biasRegularization, -1.0, factorRegularization,
                seedScale, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        // No negative factor regularization.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, -1.0,
                seedScale, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        // No negative seed scale.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                -1.0, maxIterations, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        
        // No negative max iterations.
        exceptionThrown = false;
        try
        {
            instance = new FactorizationMachinePairwiseStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, -1, random);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
    }
    
    /**
     * Test of learn method, of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testLearn()
    {
        System.out.println("learn");
        int n = 400;
        int m = 5;
        int d = 5;
        int k = 2;
        FactorizationMachine actual = new FactorizationMachine(d, k);
        actual.setBias(this.random.nextGaussian() * 10.0);
        actual.setWeights(VectorFactory.getDenseDefault().createUniformRandom(d,
            -1.0, 1.0, this.random));
        actual.setFactors(MatrixFactory.getDenseDefault().createUniformRandom(k,
            d, -1.0, 1.0, this.random));
        
        int trainSize = n;
        int testSize = n;
        int totalSize = trainSize + testSize;
        List<List<InputOutputPair<Vector, Double>>> trainData = new ArrayList<>(trainSize);
        List<List<InputOutputPair<Vector, Double>>> testData = new ArrayList<>(testSize);
        
        for (int i = 0; i < totalSize; i++)
        {
            List<InputOutputPair<Vector, Double>> list = new ArrayList<>(m);
            double offset = this.random.nextGaussian() * 100.0;
            for (int j = 0; j < m; j++)
            {
                Vector input = VectorFactory.getDenseDefault().createUniformRandom(
                    d, -10.0, 10.0, this.random);
                list.add(DefaultInputOutputPair.create(input, 
                        offset + actual.evaluateAsDouble(input)));
            }
            
            if (i < trainSize)
            {
                trainData.add(list);
            }
            else
            {
                testData.add(list);
            }
        }
        
        FactorizationMachinePairwiseStochasticGradient instance =
            new FactorizationMachinePairwiseStochasticGradient();
        instance.setFactorCount(k);
        instance.setSeedScale(0.2);
        instance.setBiasRegularization(0.0);
        instance.setWeightRegularization(0.0);
        instance.setFactorRegularization(0.1);
        instance.setLearningRate(0.005);
        instance.setMaxIterations(100);
        instance.setRandom(random);
        
//        instance.addIterativeAlgorithmListener(new IterationMeasurablePerformanceReporter());
        instance.addIterativeAlgorithmListener(new AbstractIterativeAlgorithmListener()
        {

            @Override
            public void stepEnded(IterativeAlgorithm algorithm)
            {
                final FactorizationMachinePairwiseStochasticGradient a = (FactorizationMachinePairwiseStochasticGradient) algorithm;
                MeanSquaredErrorEvaluator<Vector> performance =
                    new MeanSquaredErrorEvaluator<Vector>();
                System.out.println("Iteration " + a.getIteration() 
                    + " FCP: Train: " + NUMBER_FORMAT.format(fcp(trainData, a.getResult()))
                    + " Validation: " + NUMBER_FORMAT.format(fcp(testData, a.getResult()))
                    + " Objective: " + NUMBER_FORMAT.format(a.getObjective())
                    + " Change: " + NUMBER_FORMAT.format(a.getTotalChange())
                    + " Error: " + NUMBER_FORMAT.format(a.getEstimatedError() / a.getData().size())
                    + " Regularization: " + NUMBER_FORMAT.format(a.getRegularizationPenalty()));
            }
            
        });
        
// TODO: Figure out why this doesn't work sometimes with real factors. Is it just learning rate?
        FactorizationMachine result = instance.learn(new DefaultMultiCollection<>(trainData));
        assertEquals(d, result.getInputDimensionality());
        assertEquals(k, result.getFactorCount());
        
        System.out.println(actual.getBias());
        System.out.println(actual.getWeights());
        System.out.println(actual.getFactors());
        System.out.println(result.getBias());
        System.out.println(result.getWeights());
        System.out.println(result.getFactors());
        
        double fcp = fcp(testData, result);
        System.out.println("FCP: " + fcp);
        assertTrue(fcp >= 0.95);

    }
    
    public static double fcp(
        List<List<InputOutputPair<Vector, Double>>> testData,
        FactorizationMachine result)
    {
        int concordantPairs = 0;
        int pairs = 0;
        for (List<InputOutputPair<Vector, Double>> list : testData)
        {
            int listSize = list.size();
            double[] scores = new double[listSize];
            double[] labels = new double[listSize];
            for (int i = 0; i < listSize; i++)
            {
                scores[i] = result.evaluateAsDouble(list.get(i).getInput());
                labels[i] = list.get(i).getOutput();
            }
            
            for (int i = 0; i < listSize; i++)
            {
                for (int j = i + 1; i < listSize; i++)
                {
                    if ((labels[i] < labels[j]) == (scores[i] < scores[j]))
                    {
                        concordantPairs++;
                    }
                    pairs++;
                }
            }
        }
        return (double) concordantPairs / pairs;
    }
    /**
     * Test of learn method, of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testLearnLinear()
    {
        System.out.println("learnLinear");
        int n = 400;
        int m = 5;
        int d = 5;
        int k = 2;
        FactorizationMachine actual = new FactorizationMachine(d, k);
        actual.setBias(this.random.nextGaussian() * 10.0);
        actual.setWeights(VectorFactory.getDenseDefault().createUniformRandom(d,
            -1.0, 1.0, this.random));
        actual.setFactors(MatrixFactory.getDenseDefault().createUniformRandom(k,
            d, -1.0, 1.0, this.random));
        
        int trainSize = n;
        int testSize = n;
        int totalSize = trainSize + testSize;
        List<List<InputOutputPair<Vector, Double>>> trainData = new ArrayList<>(trainSize);
        List<List<InputOutputPair<Vector, Double>>> testData = new ArrayList<>(testSize);
        
        for (int i = 0; i < totalSize; i++)
        {
            List<InputOutputPair<Vector, Double>> list = new ArrayList<>(m);
            double offset = this.random.nextGaussian() * 100.0;
            for (int j = 0; j < m; j++)
            {
                Vector input = VectorFactory.getDenseDefault().createUniformRandom(
                    d, -10.0, 10.0, this.random);
                list.add(DefaultInputOutputPair.create(input, 
                        offset + actual.evaluateAsDouble(input)));
            }
            
            if (i < trainSize)
            {
                trainData.add(list);
            }
            else
            {
                testData.add(list);
            }
        }
        
        FactorizationMachinePairwiseStochasticGradient instance =
            new FactorizationMachinePairwiseStochasticGradient();
        instance.setFactorCount(0);
        instance.setSeedScale(0.2);
        instance.setBiasRegularization(0.0);
        instance.setWeightRegularization(0.01);
        instance.setFactorRegularization(0);
        instance.setLearningRate(0.1);
        instance.setMaxIterations(100);
        instance.setRandom(random);
        
//        instance.addIterativeAlgorithmListener(new IterationMeasurablePerformanceReporter());
        instance.addIterativeAlgorithmListener(new AbstractIterativeAlgorithmListener()
        {

            double lastObjective = Double.POSITIVE_INFINITY;
            double lastRegularization = Double.POSITIVE_INFINITY;
            @Override
            public void stepEnded(IterativeAlgorithm algorithm)
            {
                final FactorizationMachinePairwiseStochasticGradient a = (FactorizationMachinePairwiseStochasticGradient) algorithm;
                
                double objective = a.computeObjective();
                double regularization = a.getRegularizationPenalty();
                
                System.out.println("Iteration " + a.getIteration() 
                    + " FCP: Train: " + NUMBER_FORMAT.format(fcp(trainData, a.getResult()))
                    + " Validation: " + NUMBER_FORMAT.format(fcp(testData, a.getResult()))
                    + " Objective: " + NUMBER_FORMAT.format(objective)
                    + " Change: " + NUMBER_FORMAT.format(a.getTotalChange())
                    + " Error: " + NUMBER_FORMAT.format(a.getEstimatedError() / a.getData().size())
                    + " Regularization: " + NUMBER_FORMAT.format(regularization));
//                assertTrue(objective <= lastObjective);
                this.lastObjective = objective;
                this.lastRegularization = regularization;
            }
            
        });
        
// TODO: Figure out why this doesn't work sometimes with real factors. Is it just learning rate?
        FactorizationMachine result = instance.learn(new DefaultMultiCollection<>(trainData));
        assertEquals(d, result.getInputDimensionality());
        assertEquals(k, result.getFactorCount());
        
        System.out.println(actual.getBias());
        System.out.println(actual.getWeights());
        System.out.println(actual.getFactors());
        System.out.println(result.getBias());
        System.out.println(result.getWeights());
        System.out.println(result.getFactors());
        
        double fcp = fcp(testData, result);
        System.out.println("FCP: " + fcp);
        assertTrue(fcp >= 0.95);

    }

    /**
     * Test of getPerformance method, of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testGetPerformance()
    {
        FactorizationMachinePairwiseStochasticGradient instance =
            new FactorizationMachinePairwiseStochasticGradient();
        NamedValue<? extends Number> result = instance.getPerformance();
        assertEquals("objective", result.getName());
        assertEquals(0.0, result.getValue());
    }

    /**
     * Test of getLearningRate method, of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testGetLearningRate()
    {
        this.testSetLearningRate();
    }

    /**
     * Test of setLearningRate method, of class FactorizationMachinePairwiseStochasticGradient.
     */
    @Test
    public void testSetLearningRate()
    {
        double learningRate = FactorizationMachinePairwiseStochasticGradient.DEFAULT_LEARNING_RATE;
        FactorizationMachinePairwiseStochasticGradient instance =
            new FactorizationMachinePairwiseStochasticGradient();
        assertEquals(learningRate, instance.getLearningRate(), 0.0);
        
        learningRate = 0.2;
        instance.setLearningRate(learningRate);
        assertEquals(learningRate, instance.getLearningRate(), 0.0);
        
        double[] badValues = {0.0, -0.1, -2.2, Double.NaN };
        for (double badValue : badValues)
        {
            boolean exceptionThrown = false;
            try
            {
                instance.setLearningRate(badValue);
            }
            catch (IllegalArgumentException e)
            {
                exceptionThrown = true;
            }
            finally
            {
                assertTrue(exceptionThrown);
            }
            assertEquals(learningRate, instance.getLearningRate(), 0.0);
        }
    }
    
}
