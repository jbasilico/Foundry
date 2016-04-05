/*
 * File:            FactorizationMachine.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2013 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.annotation.PublicationReference;
import gov.sandia.cognition.annotation.PublicationReferences;
import gov.sandia.cognition.annotation.PublicationType;
import gov.sandia.cognition.learning.algorithm.gradient.ParameterGradientEvaluator;
import gov.sandia.cognition.learning.function.regression.AbstractRegressor;
import gov.sandia.cognition.math.DifferentiableUnivariateScalarFunction;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorEntry;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.VectorInputEvaluator;
import gov.sandia.cognition.util.ObjectUtil;

/**
 * Implements a Factorization Machine. It implements a model of pairwise 
 * interactions between features by a reduced-rank approximation. It also 
 * includes a standard linear term and a bias term.
 * 
 * The model is of the form:
 *   f(x) = b + w * x + \sum_{i=1}^{d} \sum_{j=i+1}^{d} x_i * x_j * v_i * v_j
 * where b is the bias, w is the d-dimensional weight vector and v_i are 
 * k-dimensional factor vectors for the pairwise components.
 * 
 * Factorization Machines allow for different types of Matrix Factorization
 * algorithms to be implemented as feature encodings used as input to a
 * Factorization Machine. As such, it unifies a family of algorithms that are
 * traditionally used in Recommendation Systems under an interface that looks
 * like a traditional Machine Learning algorithm.
 * 
 * @author  Justin Basilico
 * @since   3.4.0
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
        url="http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf")
})
public class FactorizationMachine
    extends AbstractRegressor<Vector>
    implements VectorInputEvaluator<Vector, Double>, 
        ParameterGradientEvaluator<Vector, Double, Vector>
{
    
    /** The bias term (b). */
    protected double bias;
    
    /** The weight vector (w) for each dimension. */
    protected Vector weights;
    
    /** The k x d factor matrix (v) with k factors for each dimension.
     *  May be null. */
    protected Matrix factors;
    
    /** Activation function for output. Null means it is linear. */
    protected DifferentiableUnivariateScalarFunction activationFunction;
    
    /**
     * Creates a new, empty {@link FactorizationMachine}. It is initialized
     * with a bias of zero and no weight or factors.
     */
    public FactorizationMachine()
    {
        this(0.0, null, null);
    }
    
    /**
     * Creates a new, empty {@link FactorizationMachine} of the given
     * input dimensionality (d) and factor count (k). It initializes the
     * internal d-dimensional weight vector and k-by-d factor matrix based on 
     * these sizes. All values are initialized to 0.
     * 
     * @param dimensionality
     *      The input dimensionality (d). Cannot be negative.
     * @param factorCount 
     *      The number of factors for pairwise interactions (k). Cannot be 
     *      negative.
     */
    public FactorizationMachine(
        final int dimensionality,
        final int factorCount)
    {
        this(0.0, VectorFactory.getDenseDefault().createVector(dimensionality),
            MatrixFactory.getDenseDefault().createMatrix(
                factorCount, dimensionality));
    }
    
    /**
     * Creates a new {@link FactorizationMachine} with the given parameters.
     * 
     * @param   bias
     *      The bias value.
     * @param   weights
     *      The weight vector of dimensionality d. May be null.
     * @param   factors 
     *      The k-by-d pairwise factor matrix. May be null.
     */
    public FactorizationMachine(
        final double bias,
        final Vector weights,
        final Matrix factors)
    {
        this(bias, weights, factors, null);
    }
    
    /**
     * Creates a new {@link FactorizationMachine} with the given parameters.
     * 
     * @param   bias
     *      The bias value.
     * @param   weights
     *      The weight vector of dimensionality d. May be null.
     * @param   factors 
     *      The k-by-d pairwise factor matrix. May be null.
     * @param   activationFunction 
     *      The activation function for the output. Can be null for a linear
     *      activation.
     */
    public FactorizationMachine(
        final double bias,
        final Vector weights,
        final Matrix factors,
        final DifferentiableUnivariateScalarFunction activationFunction)
    {
        super();
        
        this.setBias(bias);
        this.setWeights(weights);
        this.setFactors(factors);
        this.setActivationFunction(activationFunction);
    }
    
    @Override
    public FactorizationMachine clone()
    {
        final FactorizationMachine clone = (FactorizationMachine) super.clone();
        clone.weights = ObjectUtil.cloneSafe(this.weights);
        clone.factors = ObjectUtil.cloneSafe(this.factors);
        clone.activationFunction = ObjectUtil.cloneSmart(this.activationFunction);
        return clone;
    }
    
    @Override
    public double evaluateAsDouble(
        final Vector input)
    {
        double result = this.evaluateWithoutActivation(input);
        
        if (this.activationFunction != null)
        {
            // Apply the activation function.
            result = this.activationFunction.evaluate(result);
        }
        // else - Use a linear function which is no transformation.
        
        return result;
    }

    /**
     * Evaluates the factorization machine without applying the activation
     * function to the output.
     * 
     * @param   input
     *      The input value.
     * @return 
     *      The output of the factorization machine before applying the
     *      activation function.
     */
    public double evaluateWithoutActivation(
        final Vector input)
    {
        double result = this.bias;
        
        if (this.weights != null)
        {
            result += input.dot(this.weights);
        }
        // else - No weights.
        
        if (this.factors != null)
        {
            // We loop over k to do the performance improvement trick that
            // allows O(kd) computation instead of O(kd^2).
            final int factorCount = this.getFactorCount();
            final FactorActivation activation = new FactorActivation(
                input, this.factors);
            for (int k = 0; k < factorCount; k++)
            {
                // This is a faster way of looping through the non-zeros in
                // the input. It is equivalent to the code below.
                result += activation.activation(k);
                // double sum = 0.0;
                // double sumSquares = 0.0;
                // for (final VectorEntry entry : input)
                // {
                //    final double product = entry.getValue()
                //        * this.factors.getElement(k, entry.getIndex());
                //    sum += product;
                //    sumSquares += product * product;
                // }
                // result += 0.5 * (sum * sum - sumSquares);
            }
        }
        // else - No factors.
        
        return result;
    }

    @Override
    public int getInputDimensionality()
    {
        if (this.weights != null)
        {
            return this.weights.getDimensionality();
        }
        else if (this.factors != null)
        {
            return this.factors.getNumColumns();
        }
        else
        {
            // No input.
            return 0;
        }
    }

    /**
     * Gets the number of factors in the model.
     * 
     * @return 
     *      The number of factors. Cannot be negative.
     */
    public int getFactorCount()
    {
        return this.factors == null ? 0 : this.factors.getNumRows();
    }
    
    @Override
    public Vector computeParameterGradient(
        final Vector input)
    {
        Vector result = this.computeParameterGradientWithoutActivation(input);
        
        if (this.activationFunction != null)
        {
            // Differentiate the activation function.
            final double scalar = this.activationFunction.differentiate(
                this.evaluateWithoutActivation(input));
            result.scaleEquals(scalar);
        }
        
        return result;
    }
    
    /**
     * Computes the parameter gradient without applying the activation function.
     * 
     * @param   input
     *      The input vector.
     * @return 
     *      The parameter gradients without applying the activation function
     *      to them.
     */
    public Vector computeParameterGradientWithoutActivation(
        final Vector input)
    {
        final int d = this.getInputDimensionality();
        input.assertDimensionalityEquals(d);
        
        final Vector gradient = VectorFactory.getSparseDefault().createVector(
            this.getParameterCount());
        
        // The gradient for the bias is 1.
        gradient.setElement(0, 1.0);
        
        int offset = 1;
        if (this.hasWeights())
        {
            // The gradients for the linear terms are just the values from the
            // input.
            for (final VectorEntry entry : input)
            {
                gradient.setElement(offset + entry.getIndex(), entry.getValue());
            }
            offset += d;
        }
        
        if (this.hasFactors())
        {   
            // Compute the gradients per factor.
            final int factorCount = this.getFactorCount();
            for (int k = 0; k < factorCount; k++)
            {
                double sum = 0.0;
                for (final VectorEntry entry : input)
                {
                    sum += entry.getValue() * this.factors.getElement(k, 
                        entry.getIndex());
                }

                for (final VectorEntry entry : input)
                {
                    final int index = entry.getIndex();
                    final double value = entry.getValue();
                    final double factorElement = this.factors.getElement(k, index);
                    gradient.setElement(offset + index, 
                        value * (sum - value * factorElement));
                }
                
                offset += d;
            }
        }
        
        return gradient;
    }

    @Override
    public Vector convertToVector()
    {
        final int d = this.getInputDimensionality();
        
        final Vector result = VectorFactory.getSparseDefault().createVector(
            this.getParameterCount());
        result.setElement(0, this.bias);
        int offset = 1;
        if (this.hasWeights())
        {
            // Sparse iteration.
            for (final VectorEntry entry : this.weights)
            {
                result.setElement(offset + entry.getIndex(), entry.getValue());
            }
            
            offset += d;
        }
        
        if (this.hasFactors())
        {   
            // Stack factors as sparse row-wise.
            final int factorCount = this.getFactorCount();
            for (int k = 0; k < factorCount; k++)
            {
                // Sparse iteration.
                for (final VectorEntry entry : this.factors.getRow(k))
                {
                    result.setElement(offset + entry.getIndex(), entry.getValue());
                }
                
                offset += d;
            }
        }
        
        return result;
    }

    @Override
    public void convertFromVector(
        final Vector parameters)
    {
        parameters.assertDimensionalityEquals(this.getParameterCount());
        final int d = this.getInputDimensionality();
        
        // Get the bias.
        this.setBias(parameters.getElement(0));
        
        int offset = 1;
        if (this.hasWeights())
        {
            // Set the weights.
            this.setWeights(parameters.subVector(offset, offset + d - 1));
            offset += d;
        }
        
        if (this.hasFactors())
        {
            final int factorCount = this.getFactorCount();
            
            // Extract the factors for each row.
            for (int k = 0; k < factorCount; k++)
            {
                this.factors.setRow(k, 
                    parameters.subVector(offset, offset + d - 1));
                offset += d;
            }
        }
    }
    
    /**
     * Gets the number of parameters for this factorization machine. This is
     * the size of the parameter vector returned by convertToVector(). This
     * is not the number of factors (which is getFactorCount()) or the
     * size of the input dimensionality (which is getInputDimensionality()).
     * 
     * @return 
     *      The number of parameters representing this factorization machine.
     *      It is 1 plus the size of the weight vector (if there is one)
     *      plus the size of the factors matrix (if there is one).
     */
    public int getParameterCount()
    {
        final int d = this.getInputDimensionality();
        int size = 1;
        if (this.hasWeights())
        {
            size += d;
        }
        if (this.hasFactors())
        {
            size += d * this.getFactorCount();
        }
        return size;
    }
    
    /**
     * Determines if this Factorization Machine has a linear weight term.
     * 
     * @return 
     *      True if this has a linear weight term; otherwise, false.
     */
    public boolean hasWeights()
    {
        return this.weights != null;
    }
    
    /**
     * Determines if this Factorization Machine has pairwise factor terms.
     * Without pairwise factor terms, this becomes a linear model.
     * 
     * @return 
     *      True if this has pairwise factor terms; otherwise, false.
     */
    public boolean hasFactors()
    {
        return this.factors != null && this.factors.getNumRows() > 0;
    }

    /**
     * Gets the bias value.
     * 
     * @return 
     *      The bias value (b) of the model.
     */
    public double getBias()
    {
        return this.bias;
    }

    /**
     * Sets the bias value.
     * 
     * @param   bias 
     *      The bias value (b) of the model.
     */
    public void setBias(
        final double bias)
    {
        this.bias = bias;
    }

    /**
     * Gets the weight vector. It represents the linear term in the model
     * equation.
     * 
     * @return 
     *      The weight vector. May be null.
     */
    public Vector getWeights()
    {
        return this.weights;
    }

    /**
     * Sets the weight vector. It represents the linear term in the model
     * equation.
     * 
     * @param   weights 
     *      The weight vector. May be null.
     */
    public void setWeights(
        final Vector weights)
    {
        this.weights = weights;
    }

    /**
     * Gets the matrix of factors. It represents the pairwise terms in the
     * model.
     * 
     * @return 
     *      The matrix of pairwise factors. May be null.
     */
    public Matrix getFactors()
    {
        return this.factors;
    }

    /**
     * Sets the matrix of factors. It represents the pairwise terms in the
     * model.
     * 
     * @param   factors 
     *      The matrix of pairwise factors. May be null.
     */
    public void setFactors(
        final Matrix factors)
    {
        this.factors = factors;
    }

    /**
     * Gets the activation function to use for the output value. Null means 
     * linear activation.
     * 
     * @return 
     *      The activation function to use. Null means linear activation.
     */
    public DifferentiableUnivariateScalarFunction getActivationFunction()
    {
        return this.activationFunction;
    }

    /**
     * Sets the activation function to use for the output value. Null means 
     * linear activation.
     * 
     * @param   activationFunction 
     *      The activation function to use. Null means linear activation.
     */
    public void setActivationFunction(
        final DifferentiableUnivariateScalarFunction activationFunction)
    {
        this.activationFunction = activationFunction;
    }
    
    /**
     * Implementation class to compute the activation of a factor in a callback
     * driven way. This makes it efficient for looping over both dense and
     * sparse vectors.
     * 
     * This class is not thread-safe as it uses internal fields for state.
     */
    private static class FactorActivation
        implements Vector.IndexValueConsumer
    {
        /** The input vector to compute the activations for. */
        private final Vector input;
        
        /** The k-by-d matrix of factors. */
        private final Matrix factors;
        
        /** The current factor index. */
        private int factorIndex;

        /** The current sum of the individual activations along each dimension.
         */
        private double sum;

        /** The sum of the squared activations across each dimension. */
        private double sumOfSquares;
        
        /**
         * Creates a new {@link FactorActivation}.
         * 
         * @param   input
         *      The input vector to compute the activations for.
         * @param   factors 
         *      The k-by-d matrix of factors.
         */
        public FactorActivation(
            final Vector input,
            final Matrix factors)
        {
            this.input = input;
            this.factors = factors;
        }

        /**
         * Computes the activation for a given factor.
         *
         * @param factorIndex 
         *      Index for the factor to compute.
         * @return 
         *      The activation for the factor.
         */
        public double activation(
            final int factorIndex)
        {
            // Initialize the internal state.
            this.factorIndex = factorIndex;
            this.sum = 0.0;
            this.sumOfSquares = 0.0;

            // Compute the sum and sum of squares for the value along each
            // dimension. Here we use the callback and update the fields.
            this.input.forEachEntry(this);

            // Return the final result based on the sums.
            return 0.5 * (this.sum * this.sum - this.sumOfSquares);
        }

        @Override
        public void consume(
            final int featureIndex,
            final double inputValue)
        {
            // This call-back is used in the loop of activation.
            final double value = inputValue * this.factors.get(
                this.factorIndex, featureIndex);
            this.sum += value;
            this.sumOfSquares += value * value;
        }
    }
}
