/*
 * File:            ParallelFactorizationMachineStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.ParallelUtil;
import java.util.Random;
import java.util.concurrent.ThreadPoolExecutor;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for class {@link ParallelFactorizationMachineStochasticGradient}.
 * @author  Justin Basilico
 * @since   3.4.2
 */
public class ParallelFactorizationMachineStochasticGradientTest
    extends FactorizationMachineStochasticGradientTest
{
    
    /**
     * Creates a new test.
     */
    public ParallelFactorizationMachineStochasticGradientTest()
    {
        super();
    }
    
    @Test
    @Override
    public void testConstructors()
    {
        int factorCount = FactorizationMachineStochasticGradient.DEFAULT_FACTOR_COUNT;
        double learningRate = FactorizationMachineStochasticGradient.DEFAULT_LEARNING_RATE;
        double biasRegularization = FactorizationMachineStochasticGradient.DEFAULT_BIAS_REGULARIZATION;
        double weightRegularization = FactorizationMachineStochasticGradient.DEFAULT_WEIGHT_REGULARIZATION;
        double factorRegularization = FactorizationMachineStochasticGradient.DEFAULT_FACTOR_REGULARIZATION;
        double seedScale = FactorizationMachineStochasticGradient.DEFAULT_SEED_SCALE;
        int maxIterations = FactorizationMachineStochasticGradient.DEFAULT_MAX_ITERATIONS;
        ParallelFactorizationMachineStochasticGradient instance =
            new ParallelFactorizationMachineStochasticGradient();
        assertEquals(factorCount, instance.getFactorCount());
        assertEquals(learningRate, instance.getLearningRate(), 0.0);
        assertEquals(biasRegularization, instance.getBiasRegularization(), 0.0);
        assertEquals(weightRegularization, instance.getWeightRegularization(), 0.0);
        assertEquals(factorRegularization, instance.getFactorRegularization(), 0.0);
        assertEquals(seedScale, instance.getSeedScale(), 0.0);
        assertEquals(maxIterations, instance.getMaxIterations());
        assertNotNull(instance.getRandom());
        assertSame(instance.getRandom(), instance.getRandom());
        assertNotNull(instance.getThreadPool());
        assertSame(instance.getThreadPool(), instance.getThreadPool());
        
        factorCount = 22;
        learningRate = 0.12321;
        biasRegularization = 3.33;
        weightRegularization = 44.44;
        factorRegularization = 555.55;
        seedScale = 0.6;
        maxIterations = 777;
        Random random = new Random();
        ThreadPoolExecutor threadPool = ParallelUtil.createThreadPool(5);
        instance = new ParallelFactorizationMachineStochasticGradient(factorCount,
            learningRate, biasRegularization, weightRegularization, factorRegularization,
            seedScale, maxIterations, random, threadPool);
        assertEquals(factorCount, instance.getFactorCount());
        assertEquals(biasRegularization, instance.getBiasRegularization(), 0.0);
        assertEquals(weightRegularization, instance.getWeightRegularization(), 0.0);
        assertEquals(factorRegularization, instance.getFactorRegularization(), 0.0);
        assertEquals(seedScale, instance.getSeedScale(), 0.0);
        assertEquals(maxIterations, instance.getMaxIterations());
        assertSame(random, instance.getRandom());
        assertSame(threadPool, instance.getThreadPool());
        
        // No negative factor counts.
        boolean exceptionThrown = false;
        try
        {
            instance = new ParallelFactorizationMachineStochasticGradient(-1, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, 0,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, learningRate,
                -1.0, weightRegularization, factorRegularization,
                seedScale, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, learningRate,
                biasRegularization, -1.0, factorRegularization,
                seedScale, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, -1.0,
                seedScale, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                -1.0, maxIterations, random, threadPool);
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
            instance = new ParallelFactorizationMachineStochasticGradient(factorCount, learningRate,
                biasRegularization, weightRegularization, factorRegularization,
                seedScale, -1, random, threadPool);
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
    
    @Override
    protected ParallelFactorizationMachineStochasticGradient createInstance()
    {
        ParallelFactorizationMachineStochasticGradient result =
            new ParallelFactorizationMachineStochasticGradient();
        result.setThreadPool(ParallelUtil.createThreadPool(3));
        return result;
    }

    /**
     * Test of getThreadPool method, of class ParallelFactorizationMachineStochasticGradient.
     */
    @Test
    public void testGetThreadPool()
    {
        this.testSetThreadPool();
    }

    /**
     * Test of setThreadPool method, of class ParallelFactorizationMachineStochasticGradient.
     */
    @Test
    public void testSetThreadPool()
    {
        ParallelFactorizationMachineStochasticGradient instance = 
            new ParallelFactorizationMachineStochasticGradient();
        assertNotNull(instance.getThreadPool());
        assertSame(instance.getThreadPool(), instance.getThreadPool());
            
        ThreadPoolExecutor threadPool = ParallelUtil.createThreadPool(5);
        instance.setThreadPool(threadPool);
        assertSame(threadPool, instance.getThreadPool());
    }

    /**
     * Test of getNumThreads method, of class ParallelFactorizationMachineStochasticGradient.
     */
    @Test
    public void testGetNumThreads()
    {
        ParallelFactorizationMachineStochasticGradient instance
            = new ParallelFactorizationMachineStochasticGradient();
        assertTrue(instance.getNumThreads() >= 1);
    }
    
}
