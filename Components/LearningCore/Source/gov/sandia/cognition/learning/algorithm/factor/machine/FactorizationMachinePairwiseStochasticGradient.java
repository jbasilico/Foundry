/*
 * File:            FactorizationMachinePairwiseStochasticGradient.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.algorithm.MeasurablePerformanceAlgorithm;
import gov.sandia.cognition.annotation.PublicationReference;
import gov.sandia.cognition.annotation.PublicationType;
import gov.sandia.cognition.collection.CollectionUtil;
import gov.sandia.cognition.collection.MultiCollection;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_BIAS_REGULARIZATION;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_FACTOR_COUNT;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_FACTOR_REGULARIZATION;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_MAX_ITERATIONS;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_SEED_SCALE;
import static gov.sandia.cognition.learning.algorithm.factor.machine.AbstractFactorizationMachineLearner.DEFAULT_WEIGHT_REGULARIZATION;
import gov.sandia.cognition.learning.data.DatasetUtil;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.learning.function.scalar.SigmoidFunction;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.TwoVectorEntry;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorEntry;
import gov.sandia.cognition.math.matrix.VectorUnionIterator;
import gov.sandia.cognition.util.ArgumentChecker;
import gov.sandia.cognition.util.DefaultNamedValue;
import gov.sandia.cognition.util.NamedValue;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Implements a pairwise stochastic gradient descent (SGD) for learning a
 * Factorization Machine. The pairwise objective is based on the Bayesian
 * Personalized Ranking (BPR) algorithm.
 * 
 * @author  Justin Basilico
 * @since   3.4.1
 * @see     FactorizationMachine
 */
@PublicationReference(
  title="BPR: Bayesian personalized ranking from implicit feedback",
  author={"Steffen Rendle", "Christoph Freudenthaler", "Zeno Gantner", "Lars Schmidt-Thieme"},
  year=2009,
  type=PublicationType.Conference,
  publication="Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI)",
  pages={452, 461},
  url="http://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2009-Bayesian_Personalized_Ranking.pdf"
)
public class FactorizationMachinePairwiseStochasticGradient
    extends AbstractFactorizationMachineLearner
    implements MeasurablePerformanceAlgorithm
{
    
    /** The default learning rate is {@value}. */
    public static final double DEFAULT_LEARNING_RATE = 0.001;
    
    /** The learning rate for the algorithm. Must be positive. */
    protected double learningRate;

    /** The list of each list in the input data. This is used to compute and
     *  optimize the ranking objective. It is put into a list form in order to
     *  perform efficient sampling of pairs for the update steps. */
    protected List<ArrayList<? extends InputOutputPair<? extends Vector, Double>>> dataPerList;
    
    /** The total error for the current iteration. */
    protected transient double totalError;
    
    /** The total change in factorization machine parameters for the current
     *  iteration. */
    protected transient double totalChange;
    
    /**
     * Creates a new {@link FactorizationMachinePairwiseStochasticGradient} with
     * default parameters.
     */
    public FactorizationMachinePairwiseStochasticGradient()
    {
        this(DEFAULT_FACTOR_COUNT, DEFAULT_LEARNING_RATE, DEFAULT_BIAS_REGULARIZATION,
            DEFAULT_WEIGHT_REGULARIZATION, DEFAULT_FACTOR_REGULARIZATION,
            DEFAULT_SEED_SCALE, DEFAULT_MAX_ITERATIONS, new Random());
    }
    
    /**
     * Creates a new {@link FactorizationMachinePairwiseStochasticGradient}.
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
    public FactorizationMachinePairwiseStochasticGradient(
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

        // Create all the sub-lists to perform pairwise updates using.
        final MultiCollection<? extends InputOutputPair<? extends Vector, Double>> multi =
            DatasetUtil.asMultiCollection(this.data);
        this.dataPerList = new ArrayList<>(multi.getSubCollectionsCount());
        for (Collection<? extends InputOutputPair<? extends Vector, Double>> entry
            : multi.subCollections())
        {
            this.dataPerList.add(CollectionUtil.asArrayList(entry));
        }
        
        this.totalError = 0.0;
        this.totalChange = 0.0;
        return true;
    }

    @Override
    protected boolean step()
    {
        this.totalError = 0.0;
        this.totalChange = 0.0;
        
        // Sample a random list.
        for (List<? extends InputOutputPair<? extends Vector, Double>> list
            : this.dataPerList)
        {
            update(list);
        }
        
// TODO: Better convergence criteris.
        return true;
    }

    /**
     * Performs one update step for a given list. It samples a pair of examples
     * from the list that have different labels and then performs an SGD
     * update step using them.
     * 
     * @param   list
     *      The list to pull the positive and negative examples from.
     * @return 
     *      True if it was able to sample examples to do an update. False if
     *      no actual update was performed.
     */
    protected boolean update(
         final List<? extends InputOutputPair<? extends Vector, Double>> list)
    {        
        final int listSize = list.size();
        
        // Sample a random point in the list.
        final InputOutputPair<? extends Vector, Double> first = 
            list.get(this.random.nextInt(listSize));

// COULD DO: This is not an exhaustive search for something to update. Could be better by caching labels.
        // Try to find a point to actually update.
        boolean updated = false;
        for (int i = 1; i < listSize && !updated; i++)
        {        
            final InputOutputPair<? extends Vector, Double> second = 
                list.get(this.random.nextInt(listSize));
            
            double labelDiff = first.getOutput() - second.getOutput();
            if (labelDiff > 0.0)
            {
                // The first one is the preferred item.
                updated = this.update(first.getInput(), second.getInput());
            }
            else if (labelDiff < 0.0)
            {
                // The second one is the preferred item.
                updated = this.update(second.getInput(), first.getInput());
            }
            // else - Same label so keep sampling to look for one that is 
            // different.
        }
        return updated;
    }
    
    /**
     * Performs an update for the pairs of inputs where the first input
     * (positive) is preferred over the second input (negative).
     * 
     * @param   positive
     *      The first input. It is preferred to the negative.
     * @param   negative
     *      The second input. The positive is preferred to it.
     * @return 
     *      True if an update is actually made. Otherwise, false.
     */
    protected boolean update(
        final Vector positive,
        final Vector negative)
    {
        final double positiveScore = this.result.evaluate(positive);
        final double negativeScore = this.result.evaluate(negative);
        final double difference = positiveScore - negativeScore;
// TODO: Remove this debugging.
/*        
double beforeNLL = -SigmoidFunction.logistic(difference);
double beforeRegularization = this.getRegularizationPenalty();
double beforeObjective = beforeNLL
    + 0.5 * beforeRegularization;
 */
        // This term is derivative of the negative log-likelihood of the term.
        // When we try to maximize the log-likelihood ln(s(x)) = s(-x)
        // Here we use the identity that: s(x) = e^-x / (1 + e^-x).
        // However, given that we are using gradient descent, we are actually
        // minimizing the objective, which is -ln(s(x)) = -s(-x).
        final double error = -SigmoidFunction.logistic(-difference);
        
        if (error == 0.0)
        {
            // Nothing to update.
            return false;
        }
        
        this.totalError += Math.log1p(error);
        
        final double stepSize = this.learningRate;
        // No need to update bias for ranking since it cancels out.
        // this.result.setBias(0.0);
                
        if (this.isWeightsEnabled())
        {            
            // Update the weight terms.
            final Vector weights = this.result.getWeights();
            
            final VectorUnionIterator iterator = new VectorUnionIterator(
                positive, negative);
            while (iterator.hasNext())
            {
                final TwoVectorEntry entry = iterator.next();
                final int index = entry.getIndex();
                final double positiveValue = entry.getFirstValue();
                final double negativeValue = entry.getSecondValue();

                final double weightChange = stepSize * 
                    (error * (positiveValue - negativeValue)
                    + this.weightRegularization * weights.get(index));
                weights.decrement(index, weightChange);
                this.totalChange += Math.abs(weightChange);
            }
            this.result.setWeights(weights);
        }
        
        if (this.isFactorsEnabled())
        {
            // Update the factor terms.
            final Matrix factors = this.result.getFactors();

// TODO: This same calculation is needed in model evaluation.        
            for (int k = 0; k < this.factorCount; k++)
            {
                // Compute the sum of the values times the k-th factors for the
                // positive example.
                double sumPositive = 0.0;
                for (final VectorEntry entry : positive)
                {
                    sumPositive += entry.getValue() * factors.get(k, entry.getIndex());
                }
                
                // Compute the sum of the value times the k-th factors for the
                // negative example.
                double sumNegative = 0.0;
                for (final VectorEntry entry : negative)
                {
                    sumNegative += entry.getValue() * factors.get(k, entry.getIndex());
                }

                // Go over the union of what entries are in either the positive
                // or negative entries.
                final VectorUnionIterator iterator = new VectorUnionIterator(
                    positive, negative);
                while (iterator.hasNext())
                {
                    final TwoVectorEntry entry = iterator.next();
                    final int index = entry.getIndex();
                    final double positiveValue = entry.getFirstValue();
                    final double negativeValue = entry.getSecondValue();
                    final double factorElement = factors.get(k, index);
                    
                    // Compute the two gradients.
                    final double positiveGradient = 
                        positiveValue * 
                        (sumPositive - positiveValue * factorElement);
                    final double negativeGradient = 
                        negativeValue * 
                        (sumNegative - negativeValue * factorElement);
                    
                    // Update the factor based on the gradients.
                    final double factorChange = stepSize *
                        (error * (positiveGradient - negativeGradient)
                        + this.factorRegularization * factorElement);
                    factors.decrement(k, index, factorChange);
                    this.totalChange += Math.abs(factorChange);
                }
            }
            
            this.result.setFactors(factors);
        }
// TODO: Remove this debugging.
/*
double afterNLL =  -SigmoidFunction.logistic(
            this.result.evaluate(positive) - this.result.evaluate(negative));
double afterRegularization = this.getRegularizationPenalty();
                double afterObjective = afterNLL
                    + 0.5 * afterRegularization;
if (afterObjective >= beforeObjective) { 
System.out.println("Before: " + beforeObjective + " After: " + afterObjective + " Difference " + (beforeObjective - afterObjective)
        + " BR " + beforeRegularization + " AR " + afterRegularization
        + " BNLL " + beforeNLL + " ANLL " + afterNLL);
}
 */
        return true;
    }
        
    @Override
    protected void cleanupAlgorithm()
    {
        this.dataPerList = null;
    }
    
    /**
     * Computes the negative log-likelihood, which is the main part of the 
     * minimization objective. The likelihood looks at all pairs with a 
     * preference in each list in the dataset and then computes the likelihood 
     * of having the pair in the proper order by applying a logistic sigmoid to
     * the difference in scores.
     * 
     * @return 
     *      The negative log likelihood.
     */
    public double computeNegativeLogLikelihood()
    {   
        double sum = 0;
        long pairCount = 0;   
        for (ArrayList<? extends InputOutputPair<? extends Vector, Double>> list
            : this.dataPerList)
        {
            // First get all the scores and labels for the items in the list.
            final int listSize = list.size();
            final double[] scores = new double[listSize];
            final double[] labels = new double[listSize];
            for (int i = 0; i < listSize; i++)
            {
                final InputOutputPair<? extends Vector, Double> entry = list.get(i);
                scores[i] = this.result.evaluateAsDouble(entry.getInput());
                labels[i] = entry.getOutput();
            }
            
            // Now go through all the pairs of items in the list and compute
            // the log-likelihood of their swap.
// TODO: In cases where there are lots of tied labels, this loop could be optimized.
            for (int i = 0; i < listSize; i++)
            {
                final double scoreI = scores[i];
                final double labelI = labels[i];
                
                // Here we start with i+1 to only look at unique pairs.
                for (int j = i + 1; j < listSize; j++)
                {
                    final double scoreJ = scores[j];
                    final double labelJ = labels[j];
                    
                    if (labelI > labelJ)
                    {
                        // Here is i preferred to j.
                        sum += Math.log(SigmoidFunction.logistic(scoreI - scoreJ));
                        pairCount++;
                    }
                    else if (labelI < labelJ)
                    {
                        // Here j is preferred to i.
                        sum += Math.log(SigmoidFunction.logistic(scoreJ - scoreI));
                        pairCount++;
                    }
                    // else - Equal labels.
                }
            }
        }
        
        return -sum / pairCount;
    }

    
    /**
     * Gets the total objective, which is the negative log-likelihood plus the 
     * regularization terms.
     * 
     * @return 
     *      The value of the optimization objective.
     */
    public double computeObjective()
    {
        return this.computeNegativeLogLikelihood()
            + 0.5 * this.getRegularizationPenalty();
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
        if (this.result == null)
        {
            return 0.0;
        }
        
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
     * Gets the estimated error for the algorithm from the latest iteration.
     * 
     * @return 
     *      The estimated error.
     */
    public double getEstimatedError()
    {
        return this.totalError / Math.max(1, CollectionUtil.size(this.dataPerList));
    }

    /**
     * Gets the estimated total objective, which is the negative log-likelihood
     * plus the regularization terms.
     * 
     * @return 
     *      The value of the optimization objective.
     */
    public double getObjective()
    {
        return this.getEstimatedError()
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
