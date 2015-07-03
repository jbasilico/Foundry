/*
 * File:            ParallelFactorizationMachineStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2014 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.ParallelAlgorithm;
import gov.sandia.cognition.algorithm.ParallelUtil;
import gov.sandia.cognition.collection.CollectionUtil;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_BIAS_REGULARIZATION;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_FACTOR_COUNT;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_FACTOR_REGULARIZATION;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_MAX_ITERATIONS;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_SEED_SCALE;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_WEIGHT_REGULARIZATION;
import static gov.sandia.cognition.learning.algorithm.factor.machine.FactorizationMachineStochasticGradient.DEFAULT_LEARNING_RATE;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.math.matrix.Vector;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Implements a parallel version of the Stochastic Gradient Descent (SGD) 
 * algorithm for learning a Factorization Machine.
 * 
 * @author  Justin Basilico
 * @since   3.4.2
 * @see     FactorizationMachineStochasticGradient
 */
public class ParallelFactorizationMachineStochasticGradient
    extends FactorizationMachineStochasticGradient
    implements ParallelAlgorithm
{
    
    /** Thread pool used to parallelize the computation. */
    private transient ThreadPoolExecutor threadPool;

    /** The list of tasks for for running each part of the data. */
    private transient List<UpdateTask> tasks;
    
    /**
     * Creates a new {@link ParallelFactorizationMachineStochasticGradient} with
     * default parameters.
     */
    public ParallelFactorizationMachineStochasticGradient()
    {
        this(DEFAULT_FACTOR_COUNT, DEFAULT_LEARNING_RATE, DEFAULT_BIAS_REGULARIZATION,
            DEFAULT_WEIGHT_REGULARIZATION, DEFAULT_FACTOR_REGULARIZATION,
            DEFAULT_SEED_SCALE, DEFAULT_MAX_ITERATIONS, new Random(), null);
    }
    
    /**
     * Creates a new {@link ParallelFactorizationMachineStochasticGradient}.
     * 
     * @param   factorCount
     *      The number of factors to use. Zero means no factors. Cannot be
     *      negative.
     * @param   learningRate
     *      The learning rate. Must be positive.
     * @param   biasRegularization
     *      The regularization term for the bias. Cannot be negative.
     * @param   weightRegularization
     *      The regularization term for the linear weights. Cannot be negative.
     * @param   factorRegularization
     *      The regularization term for the factor matrix. Cannot be negative.
     * @param   seedScale
     *      The random initialization scale for the factors.
     *      Multiplied by a random Gaussian to initialize each factor value.
     *      Cannot be negative.
     * @param   maxIterations
     *      The maximum number of iterations for the algorithm to run. Cannot
     *      be negative.
     * @param   random 
     *      The random number generator.
     * @param   threadPool
     *      The thread pool to use.
     */
    public ParallelFactorizationMachineStochasticGradient(
        final int factorCount,
        final double learningRate,
        final double biasRegularization,
        final double weightRegularization,
        final double factorRegularization,
        final double seedScale,
        final int maxIterations,
        final Random random,
        final ThreadPoolExecutor threadPool)
    {
        super(factorCount, learningRate, biasRegularization, weightRegularization,
            factorRegularization, seedScale, maxIterations, random);
        
        this.setThreadPool(threadPool);
    }
    
    @Override
    protected boolean initializeAlgorithm()
    {
        final boolean result = super.initializeAlgorithm(); //To change body of generated methods, choose Tools | Templates.
        
        if (result)
        {
            final int taskCount = this.getNumThreads();
            this.tasks = new ArrayList<>(taskCount);
            for (List<? extends InputOutputPair<? extends Vector, Double>> part :
                CollectionUtil.createSequentialPartitions(dataList, taskCount))
            {
                tasks.add(new UpdateTask(part));
            }
        }
        
        return result;
    }
    
    @Override
    protected boolean step()
    {
// TODO: Actually update the error based on the different threads.
        this.totalError = 0.0;
        this.totalChange = 0.0;
// TODO: Should there be a more general SGD harness that does permutation of
// the order and block SGD?

        try
        {
            ParallelUtil.executeInParallel(this.tasks, this);
        }
        catch (InterruptedException | ExecutionException e)
        {
            throw new RuntimeException(e);
        }
        
// TODO: Stopping conditions.
        return true;
    }

    @Override
    protected void cleanupAlgorithm()
    {
        super.cleanupAlgorithm();;
        
        this.tasks = null;
    }
    
    @Override
    public ThreadPoolExecutor getThreadPool()
    {
        if (this.threadPool == null)
        {
            this.setThreadPool(ParallelUtil.createThreadPool());
        }

        return this.threadPool;
    }

    @Override
    public void setThreadPool(
        final ThreadPoolExecutor threadPool)
    {
        this.threadPool = threadPool;
    }

    @Override
    public int getNumThreads()
    {
        return ParallelUtil.getNumThreads(this);
    }
    
    /**
     * Implements a task to run the SGD steps on part of the data.
     */
    protected class UpdateTask
        implements Callable<Void>
    {
        /** Part of the dataset for this task to run on. */
        protected Collection<? extends InputOutputPair<? extends Vector, Double>> part;

        /**
         * Creates a new {@link UpdateTask}.
         * 
         * @param   part 
         *      Part of the dataset for the task to run on.
         */
        public UpdateTask(
            final Collection<? extends InputOutputPair<? extends Vector, Double>> part)
        {
            super();
            
            this.part = part;
        }

        @Override
        public Void call()
            throws Exception
        {
            for (final InputOutputPair<? extends Vector, Double> example 
                : this.part)
            {
                update(example);
            }
            
            return null;
        }

    }
    
}
