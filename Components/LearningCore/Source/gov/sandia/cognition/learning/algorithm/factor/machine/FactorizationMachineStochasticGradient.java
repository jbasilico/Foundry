/*
 * File:            FactorizationMachineStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2013 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.MeasurablePerformanceAlgorithm;
import gov.sandia.cognition.annotation.PublicationReference;
import gov.sandia.cognition.annotation.PublicationReferences;
import gov.sandia.cognition.annotation.PublicationType;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.data.DatasetUtil;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.math.MutableDouble;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorEntry;
import gov.sandia.cognition.util.ArgumentChecker;
import gov.sandia.cognition.util.DefaultNamedValue;
import gov.sandia.cognition.util.NamedValue;
import java.util.ArrayList;
import java.util.Random;

/**
 * Implements a Stochastic Gradient Descent (SGD) algorithm for learning a
 * Factorization Machine.
 *
 * @author  Justin Basilico
 * @since   3.4.0
 * @see     FactorizationMachine
 */
@PublicationReferences(references={
    @PublicationReference(
        title="Factorization Machines",
        author={"Steffen Rendle"},
        year=2010,
        type=PublicationType.Conference,
        publication="Proceedings of the 10th IEEE International Conference on Data Mining (ICDM)",
        url="http://www.inf.uni-konstanz.de/~rendle/pdf/Rendle2010FM.pdf"),
    @PublicationReference(
        title="Factorization Machines with libFM",
        author="Steffen Rendle",
        year=2012,
        type=PublicationType.Journal,
        publication="ACM Transactions on Intelligent Systems Technology",
        url="http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf",
        notes="Algorithm 1: Stochastic Gradient Descent (SGD)")
})
public class FactorizationMachineStochasticGradient
    extends AbstractFactorizationMachineLearner
    implements MeasurablePerformanceAlgorithm
{
// TODO: Support a version that does binary categorization.
    
    /** The default learning rate is {@value}. */
    public static final double DEFAULT_LEARNING_RATE = 0.001;
    
    /** The learning rate for the algorithm. Must be positive. */
    protected double learningRate;
    
    /** The input data represented as a list for fast access. */
    protected transient ArrayList<? extends InputOutputPair<? extends Vector, Double>> dataList;
    
    /** The sum of weights across the data. For unweighted data this is the
     *  size of the data list. */
    protected transient double weightSum;
    
    /** The total error for the current iteration. */
    protected transient double totalError;
    
    /** The total change in factorization machine parameters for the current
     *  iteration. */
    protected transient double totalChange;
    
    /**
     * Creates a new {@link FactorizationMachineStochasticGradient} with
     * default parameters.
     */
    public FactorizationMachineStochasticGradient()
    {
        this(DEFAULT_FACTOR_COUNT, DEFAULT_LEARNING_RATE, DEFAULT_BIAS_REGULARIZATION,
            DEFAULT_WEIGHT_REGULARIZATION, DEFAULT_FACTOR_REGULARIZATION,
            DEFAULT_SEED_SCALE, DEFAULT_MAX_ITERATIONS, new Random());
    }
    
    /**
     * Creates a new {@link FactorizationMachineStochasticGradient}.
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
     */
    public FactorizationMachineStochasticGradient(
        final int factorCount,
        final double learningRate,
        final double biasRegularization,
        final double weightRegularization,
        final double factorRegularization,
        final double seedScale,
        final int maxIterations,
        final Random random)
    {
        super(factorCount, biasRegularization, weightRegularization,
            factorRegularization, seedScale, maxIterations, random);
        
        this.setLearningRate(learningRate);
    }
    
    @Override
    protected boolean initializeAlgorithm()
    {
        if (!super.initializeAlgorithm())
        {
            return false;
        }

        this.dataList = CollectionUtil.asArrayList(this.data);
        this.weightSum = DatasetUtil.sumWeights(this.dataList);
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
        final double rawOutput = this.result.evaluateWithoutActivation(input);
        final double prediction = this.result.activationFunction == null
            ? rawOutput 
            : this.result.activationFunction.evaluate(rawOutput);
        
// TODO: Make the objective more pluggable. Its a bit odd for cases like a 
// logistic loss to then use a squared error objective.
        final double error = prediction - label;
        final double multiplier;
        if (this.activationFunction == null)
        {
            multiplier = error;
        }
        else
        {
            multiplier = error * this.activationFunction.differentiate(rawOutput);
        }
        
        // Compute the step size for this example.
        final double stepSize = this.learningRate * weight;
        
        if (this.isBiasEnabled())
        {
            // Update the bias term.
            final double oldBias = this.result.getBias();
            final double biasChange = stepSize * (multiplier 
                + this.biasRegularization * oldBias);
            this.result.setBias(oldBias - biasChange);
            this.totalChange += Math.abs(biasChange);
        }
        
        if (this.isWeightsEnabled())
        {
            // Update the weight terms.
            final Vector weights = this.result.getWeights();
            input.forEachEntry((
                final int index, 
                final double value) ->
            {
                final double weightChange = stepSize * 
                    (multiplier * value 
                    + this.weightRegularization * weights.get(index));

                weights.decrement(index, weightChange);
                this.totalChange += Math.abs(weightChange);
            });
        }
        
        if (this.isFactorsEnabled())
        {
            // Update the factor terms.
            final Matrix factors = this.result.getFactors();

            // This is used as a container in an inner loop so created once.
            final MutableDouble sum = new MutableDouble();
            for (int k = 0; k < this.factorCount; k++)
            {
// TODO: This same calculation is needed in model evaluation.        
                // These are used to do the call-back diven computation of the
                // loops over the sparse vectors.
                final int factorIndex = k;
                sum.value = 0.0;
                input.forEachEntry((
                    final int index,
                    final double value) ->
                {
                    sum.value += value * factors.get(factorIndex, index);
                });

                input.forEachEntry((
                    final int index,
                    final double value) ->
                {
                    final double factorElement = factors.get(factorIndex, index);
                    final double gradient = value * (sum.value - value * factorElement);
                    
                    final double factorChange = stepSize * (multiplier * gradient 
                        + this.factorRegularization * factorElement);
                    factors.decrement(factorIndex, index, factorChange);
                    this.totalChange += Math.abs(factorChange); 
                });
            }
        }
        
        this.totalError += error * error;
    }

    @Override
    protected void cleanupAlgorithm()
    {
        this.dataList = null;
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
        final double bias = this.result.getBias();
        double penalty = this.biasRegularization * bias * bias;
        
        if (this.result.hasWeights())
        {
            penalty += this.weightRegularization 
                * this.result.getWeights().norm2Squared();
        }
        
        if (this.result.hasFactors())
        {
            penalty += this.factorRegularization 
                * this.result.getFactors().normFrobeniusSquared();
        }
        
        return penalty;
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
        return 0.5 * this.getTotalError() / this.weightSum
            + 0.5 * this.getRegularizationPenalty();
    }

    @Override
    public NamedValue<? extends Number> getPerformance()
    {
        return DefaultNamedValue.create("objective", this.getObjective());
    }
    
    /**
     * Gets the learning rate. It governs the step size of the algorithm.
     * 
     * @return 
     *      The learning rate. Must be positive.
     */
    public double getLearningRate()
    {
        return this.learningRate;
    }

    /**
     * 
     * Gets the learning rate. It governs the step size of the algorithm.
     * 
     * @param learningRate 
     *      The learning rate. Must be positive.
     */
    public void setLearningRate(
        final double learningRate)
    {
        ArgumentChecker.assertIsPositive("learningRate", learningRate);
        this.learningRate = learningRate;
    }
    
}
