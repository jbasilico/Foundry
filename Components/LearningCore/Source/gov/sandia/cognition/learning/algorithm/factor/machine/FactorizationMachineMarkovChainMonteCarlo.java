/*
 * File:            FactorizationMachinePairwiseStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.annotation.PublicationReference;
import gov.sandia.cognition.annotation.PublicationReferences;
import gov.sandia.cognition.annotation.PublicationType;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorEntry;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.statistics.distribution.GammaDistribution;
import gov.sandia.cognition.statistics.distribution.UnivariateGaussian;
import gov.sandia.cognition.util.DefaultNamedValue;
import gov.sandia.cognition.util.NamedValue;
import java.util.ArrayList;
import java.util.Random;


/**
 * @TODO Document this
 * 
 * @author  Justin Basilico
 * @since   3.4.1
 * @see     FactorizationMachine
 */
@PublicationReferences(references={
    @PublicationReference(
        title="Bayesian Factorization Machines",
        author={"Christoph Freudenthaler", "Lars Schmidt-Thieme", "Steffen Rendle", "Zeno Gantner", "Christoph Freudenthaler", "Lars Schmidt-Thieme"},
        year=2011,
        type=PublicationType.Conference,
        publication="Workshop on Sparse Representation and Low-rank Approximation, Neural Information Processing Systems (NIPS-WS)",
        url="http://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_BayesianFactorizationMachines.pdf"),
    @PublicationReference(
        title="Factorization Machines with libFM",
        author="Steffen Rendle",
        year=2012,
        type=PublicationType.Journal,
        publication="ACM Transactions on Intelligent Systems Technology",
        url="http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf",
        notes="Algorithm 2: Alternating Least Squares (ALS)")
})
public class FactorizationMachineMarkovChainMonteCarlo
    extends AbstractFactorizationMachineLearner
{
    public final double DEFAULT_WEIGHT_HYPERPRIOR_MEAN = 0.0;
    public final double DEFAULT_WEIGHT_HYPERPRIOR_VARIANCE = 1.0;
    public final double DEFAULT_REGULARIZATION_HYPERPRIOR_SHAPE = 1.0;
    public final double DEFAULT_REGULARIZATION_HYPERPRIOR_RATE = 1.0;
    public final double DEFAULT_ALPHA_HYPERPRIOR_SHAPE = 1.0;
    public final double DEFAULT_ALPHA_HYPERPRIOR_RATE = 1.0;
    
    protected UnivariateGaussian weightHyperprior;
    protected GammaDistribution regularizationHyperprior;
    protected GammaDistribution alphaHyperprior;
    
    /** The size of the data. */
    protected transient int dataSize;
    
    /** The data in the form that it can be accessed in O(1) as a list. */
    protected transient ArrayList<? extends InputOutputPair<? extends Vector, Double>> dataList;
    
    /** A list representing a transposed form of the matrix of inputs. It is a
     *  d by n sparse matrix stored as an array list of sparse vectors. This is
     *  used to speed up the computation of the per-coordinate updates. */
    protected transient ArrayList<Vector> inputsTransposed;
// TODO: Should we support group regularization?    
    
    protected transient double alpha;
    protected transient double biasMu;
    protected transient double weightsMu;
    protected transient Vector factorsMu;

    protected transient double biasLambda;
    protected transient double weightsLambda;
    protected transient Vector factorsLambda;
    
    /** The current factorization machine sample. Updated in each iteration. */
    protected transient FactorizationMachine sample;
    
    protected transient FactorizationMachine cumulative;
    
    
    public FactorizationMachineMarkovChainMonteCarlo()
    {
        this(DEFAULT_FACTOR_COUNT, DEFAULT_SEED_SCALE, DEFAULT_MAX_ITERATIONS,
            new Random());
    }

    public FactorizationMachineMarkovChainMonteCarlo(
        final int factorCount,
        final double seedScale,
        final int maxIterations,
        final Random random)
    {
        super(factorCount, 0.0, 0.0,
            0.0, seedScale, maxIterations, random);
        
        this.weightHyperprior = new UnivariateGaussian(DEFAULT_WEIGHT_HYPERPRIOR_MEAN, DEFAULT_WEIGHT_HYPERPRIOR_VARIANCE);
        this.regularizationHyperprior = new GammaDistribution(DEFAULT_REGULARIZATION_HYPERPRIOR_SHAPE, 1.0 / DEFAULT_REGULARIZATION_HYPERPRIOR_RATE);
        this.alphaHyperprior = new GammaDistribution(DEFAULT_ALPHA_HYPERPRIOR_SHAPE, 1.0 / DEFAULT_ALPHA_HYPERPRIOR_RATE);
    }
    
    @Override
    protected boolean initializeAlgorithm()
    {
// TODO: This was largely copied from the ALS implementation.   
        if (!super.initializeAlgorithm())
        {
            return false;
        }
        this.dataSize = this.data.size();
        if (this.dataSize <= 0)
        {
            return false;
        }
        
        this.dataList = CollectionUtil.asArrayList(this.data);
        final VectorFactory<?> sparseFactory = VectorFactory.getSparseDefault();
        this.inputsTransposed = new ArrayList<>(this.dimensionality);
        for (int i = 0; i < this.dimensionality; i++)
        {
            this.inputsTransposed.add(sparseFactory.createVector(this.dataSize));
        }
        
        // Fill in the transposed data.
        for (int i = 0; i < this.dataSize; i++)
        {
            final InputOutputPair<? extends Vector, Double> example = 
                this.dataList.get(i);
            for (final VectorEntry entry : example.getInput())
            {
                if (entry.getValue() != 0.0)
                {
                    this.inputsTransposed.get(entry.getIndex()).setElement(i,
                        entry.getValue());
                }
            }
        }
        
        this.alpha = 0.0;        
        this.biasMu = 0.0;
        this.weightsMu = 0.0;
        this.factorsMu = VectorFactory.getDenseDefault().createVector(this.factorCount);
        this.biasLambda = 0.0;
        this.weightsLambda = 0.0;
        this.factorsLambda = VectorFactory.getDenseDefault().createVector(this.factorCount);
        
        // The result is initialized to the sample.
        this.sample = this.result;
        this.cumulative = new FactorizationMachine(this.dimensionality, this.factorCount);
        this.result = null;
        
        return true;
    }
    
    @Override
    protected boolean step()
    {
// TODO: Should errors just be computed once and then updated?
        final Vector errors = VectorFactory.getDenseDefault().createVector(
            this.dataSize);
        
        // Compute the initial prediction and error terms per input.
        for (int i = 0; i < this.dataSize; i++)
        {
            final InputOutputPair<? extends Vector, Double> example = 
                this.dataList.get(i);
            final double prediction = this.sample.evaluateAsDouble(example.getInput());
            final double actual = example.getOutput();
            final double error = actual - prediction;
            errors.setElement(i, error);
        }
        
        final double alpha0 = this.alphaHyperprior.getShape();
        final double beta0 = this.alphaHyperprior.getRate();
        final double alpha = this.sampleGamma(
            0.5 * (alpha0 + this.dataSize), 
            0.5 * (errors.norm2Squared() + beta0));
System.out.println("Errors: " + errors.norm2Squared());
System.out.println("alpha sampler: " + 
(            0.5 * (alpha0 + this.dataSize)) + " " +
            (0.5 * (errors.norm2Squared() + beta0)));
        this.alpha = alpha;
        
        // Sample the priors and regularizations.
        final double mu0 = this.weightHyperprior.getMean();
        final double gamma0 = this.weightHyperprior.getVariance();
        final double alphaLambda = this.regularizationHyperprior.getShape();
        final double betaLambda = this.regularizationHyperprior.getRate();
        final int d = this.dimensionality;
                    
        // Sample the bias.
        if (this.isBiasEnabled())
        {
            final double oldBias = this.sample.getBias();
            double biasPrior = this.biasMu;
            final double biasRegularization = this.sampleGamma(
                0.5 * (alphaLambda + 1.0 + 1.0), 
                0.5 * (gamma0 * Math.pow(biasPrior - mu0, 2) + betaLambda));
            biasPrior = this.sampleGaussian(
                (oldBias + gamma0 * mu0)/ (1.0 + gamma0),
                1.0 / ((1.0 + gamma0) * biasRegularization));
            this.biasMu = biasPrior;
            this.biasLambda = biasRegularization;

            final double variance = 1.0 / (alpha * dataSize + biasRegularization);
            final double mean = variance * (
                alpha * oldBias * dataSize 
                + alpha * errors.sum() 
                + biasPrior * biasRegularization);
            final double newBias = this.sampleGaussian(mean, variance);
            
            // Update the running errors.
            final double biasChange = oldBias - newBias;
            for (int i = 0; i < this.dataSize; i++)
            {
                errors.increment(i, biasChange);
            }
            this.sample.setBias(newBias);
            this.cumulative.setBias(this.cumulative.getBias() + newBias);
        }
        
        // Sample the weights.
        if (this.isWeightsEnabled())
        {
            final Vector weights = this.sample.getWeights();

            
            final double weightRegularization;
            double weightPrior;
            {
                final double weightSum = weights.sum();
                final double weightMean = weightSum / d;
                final double sumOfSquares =
                    weights.dotProduct(weights) - 2.0 * weights.scale(weightMean).sum()
                    + d * weightMean * weightMean;
                
                weightPrior = this.weightsMu;
                weightRegularization = this.sampleGamma(
                    0.5 * (alphaLambda + d + 1.0),
                    0.5 * (sumOfSquares + gamma0 * Math.pow(weightPrior - mu0, 2) + betaLambda));
                weightPrior = this.sampleGaussian(
                    (weightSum + gamma0 * mu0) / (d + gamma0),
                    1.0 / ((d + gamma0) * weightRegularization));
                this.weightsMu = weightPrior;
                this.weightsLambda = weightRegularization;
            }
                
            for (int j = 0; j < this.dimensionality; j++)
            {
                final double oldWeight = weights.getElement(j);
                final Vector inputs = this.inputsTransposed.get(j);
// TODO: This could be cached and computed once.
                final Vector derivative = inputs;
                final double sumOfSquares = derivative.norm2Squared();
                
                final double variance = 
                    1.0 / (alpha * sumOfSquares + weightRegularization);
                final double newWeight = this.sampleGaussian(
                    variance * (alpha * oldWeight * sumOfSquares 
                    + alpha * derivative.dotProduct(errors)
                    + weightPrior * weightRegularization),
                    variance);
// TODO: Prevent divide-by-zero. If sum of squares is zero, then weight should be 0.
                weights.setElement(j, newWeight);

                // Update the running errors.
                final double weightChange = oldWeight - newWeight;
                errors.scaledPlusEquals(weightChange, inputs);
            }
            this.sample.setWeights(weights);
            this.cumulative.getWeights().plusEquals(weights);
        }
        
        
        // Update the factors.
        if (this.isFactorsEnabled())
        {
            final Matrix factors = this.sample.getFactors();
            for (int k = 0; k < this.factorCount; k++)
            {
                final Vector factorRow = factors.getRow(k);
                
                final double factorRegularization;
                double factorPrior;
                {
                    
                    final double factorSum = factorRow.sum();
                    final double factorMean = factorSum / d;
                    final double sumOfSquares =
                        factorRow.dotProduct(factorRow) 
                        - 2.0 * factorRow.scale(factorMean).sum()
                        + d * factorMean * factorMean;

                    factorPrior = this.factorsMu.getElement(k);
                    factorRegularization = this.sampleGamma(
                        0.5 * (alphaLambda + d + 1.0),
                        0.5 * (sumOfSquares + gamma0 * Math.pow(factorPrior - mu0, 2) + betaLambda));
                    factorPrior = this.sampleGaussian(
                        (factorSum + gamma0 * mu0) / (d + gamma0),
                        1.0 / ((d + gamma0) * factorRegularization));
                    this.factorsMu.setElement(k, factorPrior);
                    this.factorsLambda.setElement(k, factorRegularization);
                }
                final Vector factorTimesInput = VectorFactory.getDefault().createVector(
                    this.dataSize);

                for (int i = 0; i < this.dataSize; i++)
                {
                    factorTimesInput.setElement(i,
                        this.dataList.get(i).getInput().dotProduct(factorRow));
                }

                for (int j = 0; j < this.dimensionality; j++)
                {
                    final double oldFactor = factorRow.getElement(j);
                    final Vector inputs = this.inputsTransposed.get(j);
                    final Vector derivative = inputs.dotTimes(factorTimesInput);
// TODO: This inputs^2 could be cached and computed once.
                    derivative.scaledMinusEquals(oldFactor, inputs.dotTimes(inputs));
                    final double sumOfSquares = derivative.norm2Squared();
// TODO: Prevent divide-by-zero either by checking denominator or forcing factor regularization to be positive.
                    final double variance = 1.0 / (alpha * sumOfSquares + factorRegularization);
                    final double mean = variance *
                        (alpha * oldFactor * sumOfSquares 
                         + alpha * derivative.dotProduct(errors)
                         + factorPrior * factorRegularization);
                    final double newFactor = this.sampleGaussian(
                        mean, variance);

                    factors.setElement(k, j, newFactor);

                    // Update the running errors and factor times input.
                    final double factorChange = oldFactor - newFactor;
                    errors.scaledPlusEquals(factorChange, derivative);
                    factorTimesInput.scaledPlusEquals(-factorChange, inputs);
                }
            }
            this.sample.setFactors(factors);
            this.cumulative.getFactors().plusEquals(factors);
        }
        
        return true;
    }

    @Override
    protected void cleanupAlgorithm()
    {
        this.dataList = null;
        this.inputsTransposed = null;
        this.sample = null;
    }
// TODO: This is an override, maybe it shouldn't be extending.
    @Override
    public FactorizationMachine getResult()
    {
        // The result is the average element from the chain.
        return new FactorizationMachine(
            this.cumulative.getBias() / this.iteration,
            this.cumulative.getWeights().scale(1.0 / this.iteration),
            this.cumulative.getFactors().scale(1.0 / this.iteration));
    }

    public FactorizationMachine getSample()
    {
        return this.sample;
    }
    
    public NamedValue<? extends Number> getPerformance()
    {
        return DefaultNamedValue.create("TEMP (zero)", 0.0);
// TODO: Implement this.
    }
    
    protected double sampleGamma(
        final double shape,
        final double rate)
    {
// TODO: Avoid boxing.
        return new GammaDistribution(shape, 1.0 / rate).sampleAsDouble(this.random);
    }
    
    protected double sampleGaussian(
        final double mean,
        final double variance)
    {
// TODO: Avoid boxing.
        return new UnivariateGaussian(mean, variance).sampleAsDouble(this.random);
    }
    
    
 
// TODO: Getters and setters for hyperpriors.

    public double getAlpha()
    {
        return alpha;
    }
    
}
