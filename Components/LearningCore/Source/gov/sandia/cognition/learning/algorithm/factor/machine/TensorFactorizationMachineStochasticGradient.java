/*
 * File:            TensorFactorizationMachineStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.MeasurablePerformanceAlgorithm;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.algorithm.AbstractAnytimeSupervisedBatchLearner;
import gov.sandia.cognition.learning.data.DatasetUtil;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.util.DefaultNamedValue;
import gov.sandia.cognition.util.NamedValue;
import gov.sandia.cognition.util.Randomized;
import java.util.ArrayList;
import java.util.Random;

/**
 * Implements the Stochastic Gradient Descent (SGD) algorithm for learning a 
 * higher-order Factorization Machine.
 * 
 * @author  Justin Basilico
 * @since   3.4.3
 * @see     TensorFactorizationMachine
 */
public class TensorFactorizationMachineStochasticGradient
    extends AbstractAnytimeSupervisedBatchLearner<Vector, Double, TensorFactorizationMachine>
    implements Randomized, MeasurablePerformanceAlgorithm
{
    
    /**
     * The default learning rate is {@value}.
     */
    public static final double DEFAULT_LEARNING_RATE = 0.001;

    /** The number of factors to do for each n-way interaction, starting with 
     *  2-way. */
    protected int[] factorCountPerWay;
    
    /** Regularization for bias term. */
    protected double biasRegularization;
    
    /** Regularization for linear terms. */
    protected double weightRegularization;
    
    /** Regularization for each way of factors. */
    protected double[] regularizationPerWay;

    /**
     * The learning rate for the algorithm. Must be positive.
     */
    protected double learningRate;
    
    /** The standard deviation for initializing the factors. Cannot be negative.
     */
    protected double[] seedScalePerWay;
    
    /** The random number generator to use. */
    protected Random random;
    
    /** The input data represented as a list for fast access. */
    protected transient ArrayList<? extends InputOutputPair<? extends Vector, Double>> dataList;
        
    /** The current factorization machine output learned by the algorithm. */
    protected transient TensorFactorizationMachine result;
    
    /** Contains regularization weights for each parameter in the parameter
     *  vector of the factorization machine. */
    protected transient Vector regularizationMask;
    
    /** The total error for the current iteration. */
    protected transient double totalError;
    
    /** The total change in factorization machine parameters for the current
     *  iteration. */
    protected transient double totalChange;

    
// TODO: Default constructor.
    
// TODO: Document this.
    public TensorFactorizationMachineStochasticGradient(
        final int[] factorCountPerWay,
        final double biasRegularization,
        final double weightRegularization,
        final double[] regularizationPerWay,
        final double learningRate,
        final double[] seedScalePerWay,
        final int maxIterations,
        final Random random)
    {
        super(maxIterations);
        
        this.factorCountPerWay = factorCountPerWay;
        this.biasRegularization = biasRegularization;
        this.weightRegularization = weightRegularization;
        this.regularizationPerWay = regularizationPerWay;
        this.learningRate = learningRate;
        this.seedScalePerWay = seedScalePerWay;
        this.random = random;
    }
    
    @Override
    protected boolean initializeAlgorithm()
    {
        // Initialize the weight vectors.
        int dimensionality = DatasetUtil.getInputDimensionality(this.data);
        
        // Initialize the factorization.
        this.result = new TensorFactorizationMachine(dimensionality, this.factorCountPerWay);
        
        int wayIndex = -1;
        for (final Matrix factors : this.result.getFactorsPerWay())
        {
            wayIndex++;
            if (factors == null)
            {
                continue;
            }

            // Initialize the factors to small random gaussian values.
            final int factorCount = factors.getNumRows();
            final double seedScale = this.seedScalePerWay[wayIndex];
            for (int i = 0; i < dimensionality; i++)
            {
                for (int j = 0; j < factorCount; j++)
                {
                    factors.setElement(j, i, 
                        seedScale * this.random.nextGaussian());
                }
            }
        }
        
        this.regularizationMask = VectorFactory.getDenseDefault().createVector(result.getParameterCount());
        this.regularizationMask.set(0, this.biasRegularization);
        int index = 1;
        for (int i = 0; i < dimensionality; i++)
        {
            this.regularizationMask.set(index, this.weightRegularization);
            index++;
        }
        for (int way = 0; way < this.factorCountPerWay.length; way++)
        {
            final int factorCount = this.factorCountPerWay[way];
            final double factorRegularization = this.regularizationPerWay[way];
            for (int i = 0; i < factorCount; i++)
            {
                for (int j = 0; j < dimensionality; j++)
                {
                    this.regularizationMask.set(index, factorRegularization);
                    index++;
                }
            }
        }
        
        this.dataList = CollectionUtil.asArrayList(this.data);
        this.totalError = 0.0;
        this.totalChange = 0.0;
        
        return true;
    }

    @Override
    protected boolean step()
    {
        this.totalError = 0.0;
        this.totalChange = 0.0;
        
// TODO: Should there be a more general SGD harness that does permutation of
// the order and block SGD?
        for (final InputOutputPair<? extends Vector, Double> example 
            : this.dataList)
        {
            this.update(example);
        }
// TODO: Stopping conditions.
        return true;
    }
    
    /**
     * Performs a single update of step of the stochastic gradient descent
     * by updating according to the given example.
     * 
     * @param   example 
     *      The example to do a stochastic gradient step for.
     */
    protected void update(
        final InputOutputPair<? extends Vector, Double> example)
    {
        final Vector input = example.getInput();
        final double label = example.getOutput();
        final double weight = DatasetUtil.getWeight(example);
        final double prediction = this.result.evaluateAsDouble(input);
        final double error = prediction - label;
        
        // Compute the step size for this example.
// TODO: Should this be total weight?
        final double stepSize = this.learningRate * weight / this.data.size();
        
        final Vector delta = this.result.computeParameterGradient(input);
        delta.scaleEquals(-stepSize * error);
        delta.scaledPlusEquals(-stepSize, 
            this.result.getActiveParameterVector(input).dotTimes(
                this.regularizationMask));
        this.result.incrementParameterVector(delta);
        
        this.totalChange += delta.norm1();
        this.totalError += error * error;
    }

    @Override
    protected void cleanupAlgorithm()
    {
        this.dataList = null;
    }
    
    @Override
    public TensorFactorizationMachine getResult()
    {
        return this.result;
    }

    /**
     * Gets the total change from the current iteration.
     * 
     * @return 
     *      The total change in the parameters of the factorization machine.
     */
    public double getTotalChange()
    {
        return this.totalChange;
    }
    
    /**
     * Gets the total squared error from the current iteration.
     * 
     * @return 
     *      The total squared error.
     */
    public double getTotalError()
    {
        return this.totalError;
    }
    
    /**
     * Gets the regularization penalty term for the current result. It
     * computes the squared 2-norm of the parameters of the factorization
     * machine, each multiplied with their appropriate regularization weight.
     * 
     * @return 
     *      The regularization penalty term for the objective.
     */
    public double getRegularizationPenalty()
    {
        final Vector parameters = this.result.convertToVector();
        return parameters.dotTimes(parameters).dot(this.regularizationMask);
    }
    
    /**
     * Gets the total objective, which is the mean squared error plus the 
     * regularization terms.
     * 
     * @return 
     *      The value of the optimization objective.
     */
    public double getObjective()
    {
// TODO: Should data size be total weight?
        return 0.5 * this.getTotalError() / this.data.size()
            + 0.5 * this.getRegularizationPenalty();
    }
    
    @Override
    public NamedValue<? extends Number> getPerformance()
    {
        return DefaultNamedValue.create("objective", this.getObjective());
    }

    @Override
    public Random getRandom()
    {
        return this.random;
    }

    @Override
    public void setRandom(
        final Random random)
    {
        this.random = random;
    }
    
}
