/*
 * File:            TensorFactorizationMachine.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2014 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.annotation.PublicationReference;
import gov.sandia.cognition.annotation.PublicationReferences;
import gov.sandia.cognition.annotation.PublicationType;
import gov.sandia.cognition.learning.algorithm.gradient.ParameterGradientEvaluator;
import gov.sandia.cognition.learning.function.regression.AbstractRegressor;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorEntry;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.VectorInputEvaluator;
import gov.sandia.cognition.util.ArgumentChecker;
import gov.sandia.cognition.util.ObjectUtil;
import java.util.Arrays;

/**
 * Implements a higher-order Factorization Machine. It models sets of n-way
 * interactions between features by a reduced-rank approximation. It also
 * includes a standard linear term and a bias term.
 * 
 * The model is of the form:
 *   f(x) = b + w * x + \sum_{l=1}^{n} \sum_{i_1=1}^{d} ... \sum_{i_l=i_{l-1} + 1} (\product_{j=1}^{l} x_{i_j}) (\sum_{f=1}^{k_l} \product_{j=1}^{l} v_{l,i_j,f})
 * where b is the bias, w is the d-dimensional weight vector and v_{l,i} are 
 * k_l-dimensional factor vectors for the l-way interactions.
 * 
 * Higher-order Factorization Machines generalize standard Factorization 
 * Machines, which model pairwise (n=2) interactions. While standard 
 * Factorization Machines generalize different types of Matrix Factorizations,
 * the higher-order Factorization Machine can generalize different types of
 * Tensor Factorizations such as PARAFAC. This is done by implementing 
 * different feature encodings, which makes these factorizations look like a
 * traditional Machine Learning algorithm. It is typically used with 
 * high-dimensional, sparse data.
 * 
 * Currently, only orders of up to 10-way interactions are supported, since
 * part of the computation for these interactions are stored in a cached table.
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
        url="http://www.csie.ntu.edu.tw/~b97053/paper/Factorization%20Machines%20with%20libFM.pdf")
})
public class TensorFactorizationMachine
    extends AbstractRegressor<Vector>
    implements VectorInputEvaluator<Vector, Double>,
        ParameterGradientEvaluator<Vector, Double, Vector>
{
    
    /** The bias term (b). */
    protected double bias;
    
    /** The weight vector (w) for each dimension. */
    protected Vector weights;
    
    /** The array of factor matrices for each way in the factorization. The
     *  ways only have factors for 2-way and higher, so the first index in the
     *  matrix are the factors for the 2-way factorization. For a given way (l),
     *  this contains k_l x d factor matrix (v) with k_l factors for each 
     *  dimension. Cannot be null. */
    protected Matrix[] factorsPerWay;

// TODO: May need to enforce that the number of ways is less than or equal to d.
    
    /**
     * Creates a new, empty {@link TensorFactorizationMachine}. It is
     * initialized with a bias of zero and no weight or factors.
     */
    public TensorFactorizationMachine()
    {
        this(0.0, null, new Matrix[0]);
    }
    
    /**
     * Creates a new, empty {@link TensorFactorizationMachine} of the given
     * input dimensionality and number of factors for each n-way interaction
     * for two and greater.
     * 
     * @param   dimensionality
     *      The input dimensionality (d). Cannot be negative.
     * @param   factorCountsPerWay 
     *      An array of k_l by d factor matrices for each way l starting from
     *      ways of 2. Cannot be null. Can be empty. Each factor
     *      matrix can be null to mean that interaction is not used.
     *      Currently, only up to 10-way interactions are supported, which 
     *      means the maximum size of the array is 8.
     */
    public TensorFactorizationMachine(
        final int dimensionality,
        final int... factorCountsPerWay)
    {
        this(0.0, VectorFactory.getDenseDefault().createVector(dimensionality),
            new Matrix[factorCountsPerWay.length]);
        
        // Initialize the matrices.
        final MatrixFactory<?> matrixFactory = MatrixFactory.getDenseDefault();
        for (int i = 0; i < factorCountsPerWay.length; i++)
        {
            final int factorCount = factorCountsPerWay[i];
            
            if (factorCount > 0)
            {
                this.factorsPerWay[i] = matrixFactory.createMatrix(
                    factorCount, dimensionality);
            }
            // else - No factors for that way.
        }
    }

    /**
     * Creates a new {@link TensorFactorizationMachine} with the given
     * parameters.
     * 
     * @param   bias
     *      The bias value.
     * @param   weights
     *      The weight vector of dimensionality d. May be null.
     * @param   factorsPerWay 
     *      An array of k_l by d factor matrices for each way l starting from
     *      pairwise interactions. Cannot be null. Can be empty. Each factor
     *      matrix can be null to mean that interaction is not used.
     */
    public TensorFactorizationMachine(
        final double bias,
        final Vector weights,
        final Matrix... factorsPerWay)
    {
        super();
        
        this.setBias(bias);
        this.setWeights(weights);
        this.setFactorsPerWay(factorsPerWay);
    }
    
    @Override
    public TensorFactorizationMachine clone()
    {
        final TensorFactorizationMachine clone = (TensorFactorizationMachine) super.clone();
        clone.weights = ObjectUtil.cloneSafe(this.weights);
        clone.factorsPerWay = ObjectUtil.cloneSmartArrayAndElements(this.factorsPerWay);
        return clone;
    }
    
    @Override
    public double evaluateAsDouble(
        final Vector input)
    {
        double result = this.bias;
        
        if (this.weights != null)
        {
            result += this.weights.dotProduct(input);
        }
        // else - No weights.
        
        // Go through all the factors for each way.
        int way = 1;
        for (final Matrix factors : this.factorsPerWay)
        {
            way++;
            if (factors == null)
            {
                // Skip empty factors.
                continue;
            }

            // We loop over k to do the performance improvement trick that
            // allows O(kd) computation instead of O(kd^l).
            final int factorCount = factors.getNumRows();
            
            // We need to compute the total sum for the way. In doing so, we
            // also will need to know the sums of the powers of each active
            // feature in each factor, which is what the array is used for.
            double waySum = 0.0;
            final double[] sums = new double[way + 1];
            final double[] partial = new double[way + 1];
            for (int k = 0; k < factorCount; k++)
            {
                // Clear the sums of the terms to each power.
                Arrays.fill(sums, 0.0);
                Arrays.fill(partial, 0.0);
                partial[0] = 1.0;
                for (final VectorEntry entry : input)
                {
                    final double product = entry.getValue()
                        * factors.getElement(k, entry.getIndex());
                    
                    // Fill in all the powers.
                    double accumulator = product;
                    for (int i = 1; i <= way; i++)
                    {
                        sums[i] += accumulator;
                        accumulator *= product;
                    }
                }
               
                for (int i = 1; i <= way; i++)
                {
                    double partialSum = 0.0;
                    int sign = 1;
                    for (int j = 1; j <= i; j++)
                    {
                        partialSum += sums[j] * partial[i - j] 
                            * sign * FACTORIAL_PER_WAY[i - 1] / FACTORIAL_PER_WAY[i - j];
                        sign *= -1;
                    }
//                    partialSum += sign * sums[i] * FACTORIAL_PER_WAY[i - 1];
                    partial[i] = partialSum;
                }
                waySum += partial[way];
            }
            
            // Update the result with the total value for this way.
            result += waySum / FACTORIAL_PER_WAY[way];
        }
        
        return result;
    }
    
    @Override
    public Vector computeParameterGradient(
        final Vector input)
    {
        final int d = this.getInputDimensionality();
        input.assertDimensionalityEquals(d);
        
        final Vector gradient = VectorFactory.getSparseDefault().createVector(
            this.getParameterCount());
        
        // The gradient for the bias is 1.
        gradient.setElement(0, 1.0);
        
        int offset = 1;
        if (this.weights != null)
        {
            // The gradients for the linear terms are just the values from the
            // input.
            for (final VectorEntry entry : input)
            {
                gradient.setElement(offset + entry.getIndex(), entry.getValue());
            }
            offset += d;
        }

        // Go through all the factors for each way.
        int way = 1;
        for (final Matrix factors : this.factorsPerWay)
        {
            way++;
            if (factors == null)
            {
                // Skip empty factors.
                continue;
            }
            
            // We loop over k to do the performance improvement trick that
            // allows O(kd) computation instead of O(kd^l).
            final int factorCount = factors.getNumRows();
            final int[][] partitions = PARTITIONS_PER_WAY[way];
            final int partitionCount = partitions.length;
            final int[] termCoefficients = COEFFICIENTS_PER_WAY[way];
            final int wayFactorial = FACTORIAL_PER_WAY[way];
            
            // We need to compute the total sum for the way. In doing so, we
            // also will need to know the sums of the powers of each active
            // feature in each factor, which is what the array is used for.
            final double[] sums = new double[way + 1];
            final double[] terms = new double[partitions.length];
            final double[] valuePowers = new double[way + 1];
            final double[] factorPowers = new double[way];
            valuePowers[0] = 1.0;
            factorPowers[0] = 1.0;
            
            for (int k = 0; k < factorCount; k++)
            {
                // Clear the sums of the terms to each power.
                Arrays.fill(sums, 0.0);
                for (final VectorEntry entry : input)
                {
                    final double product = entry.getValue()
                        * factors.getElement(k, entry.getIndex());
                    
                    // Fill in all the powers.
                    double accumulator = product;
                    for (int i = 1; i <= way; i++)
                    {
                        sums[i] += accumulator;
                        accumulator *= product;
                    }
                }
                
                // Compute each term in the expansion for this way by using
                // the partitions to figure out the powers of elements to use.
                for (int i = 0; i < partitionCount; i++)
                {
                    // Compute the value of this term using the sums of each
                    // power and then multiplying them using the defined
                    // partition.
                    double term = 1.0;
                    for (final int power : partitions[i])
                    {
                        term *= sums[power];
                    }
                    
                    // Now add to the total sum using the appropriate 
                    // coefficient.
                    terms[i] = termCoefficients[i] * term / wayFactorial;
                }
                
                for (final VectorEntry entry : input)
                {
                    final int index = entry.getIndex();
                    final double value = entry.getValue();
                    final double factorElement = factors.getElement(k, index);
// TODO: These arrays could probably be combined since they're only multiplied together
// That is, it can contain p[n] = x^n v^(n-1)
                    valuePowers[1] = value;
                    factorPowers[1] = factorElement;
                    for (int i = 2; i < way; i++)
                    {
                        valuePowers[i] = value * valuePowers[i - 1];
                        factorPowers[i] = factorElement * factorPowers[i - 1];
                    }
                    valuePowers[way] = value * valuePowers[way - 1];
                    
                    double partial = 0.0;
                    for (int i = 0; i < partitionCount; i++)
                    {
                        // Compute the value of this term using the sums of each
                        // power and then multiplying them using the defined
                        // partition.
                        final double term = terms[i];
                        if (term == 0.0)
                        {
                            // Ignore zero terms to avoid dividing by zero,
                            // since the term is the product of the sum entries
                            // so if one is zero, then the whole term is zero.
                            continue;
                        }
                        
                        for (final int power : partitions[i])
                        {
                            partial += power * factorPowers[power - 1]
                                * valuePowers[power] * term / sums[power];
                        }
                    }
                    
                    gradient.setElement(offset + index, partial);
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
        if (this.weights != null)
        {
            // Sparse iteration.
            for (final VectorEntry entry : this.weights)
            {
                result.setElement(offset + entry.getIndex(), entry.getValue());
            }
            
            offset += d;
        }
        
        for (final Matrix factors : this.factorsPerWay)
        {
            if (factors == null)
            {
                // These factors are not used.
                continue;
            }
            
            // Stack factors as sparse row-wise.
            final int factorCount = factors.getNumRows();
            for (int k = 0; k < factorCount; k++)
            {
                // Sparse iteration.
                for (final VectorEntry entry : factors.getRow(k))
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
        if (this.weights != null)
        {
            // Set the weights.
            this.setWeights(parameters.subVector(offset, offset + d - 1));
            offset += d;
        }
        
        for (final Matrix factors : this.factorsPerWay)
        {
            if (factors == null)
            {
                // These factors are not used.
                continue;
            }
            
            final int factorCount = factors.getNumRows();
            
            // Extract the factors for each row.
            for (int k = 0; k < factorCount; k++)
            {
                factors.setRow(k, 
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
     *      plus the size of all the factors matrices (if there are any).
     */
    public int getParameterCount()
    {
        final int d = this.getInputDimensionality();
        int size = 1;
        if (this.weights != null)
        {
            size += d;
        }
        
        for (final Matrix factors : this.factorsPerWay)
        {
            if (factors != null)
            {
                size += d * factors.getNumRows();
            }
        }

        return size;
    }
    
    @Override
    public int getInputDimensionality()
    {
        if (this.weights != null)
        {
            return this.weights.getDimensionality();
        }

        // Find the first non-null factors.
        for (final Matrix factors : this.factorsPerWay)
        {
            if (factors != null)
            {
                return factors.getNumColumns();
            }
        }
        
        // No input.
        return 0;
    }
    
    /**
     * Determines if this Factorization Machine is using the given way of
     * interaction. The linear term is considered way 1, and the factorized
     * interactions are for ways of 2 or greater. It always has way 0, which is
     * the bias.
     * 
     * @param   way
     *      The way.
     * @return 
     *      True if the factorization machine is modeling the given way of
     *      interaction.
     */
    public boolean hasWay(
        final int way)
    {
        if (way < 0)
        {
            // Bad input.
            return false;
        }
        else if (way == 0)
        {
            // We always have a 0-th way.
            return true;
        }
        else if (way == 1)
        {
            // See if there is a first way (the weights).
            return this.weights != null;
        }
        else if (way <= this.getMaxWay())
        {
            return this.getFactors(way) != null;
        }
        
        // Way was beyond the maximum.
        return false;
    }
    
    /**
     * Gets the maximum of valid ways in the factorization machine. There is 
     * always at least 1 way (which is the linear term), though only ways 2
     * and higher have factors.
     * @return 
     */
    public int getMaxWay()
    {
        return 1 + this.factorsPerWay.length;
    }
    
    /**
     * Gets the factors for a given way.
     * 
     * @param   way
     *      The way to get the factors for. Must be greater than 1.
     * @return 
     *      The k-by-d factors for the given way, if there are any. Otherwise,
     *      null.
     */
    public Matrix getFactors(
        final int way)
    {
        final int index = way - 2;
        if (index < 0)
        {
            throw new IllegalArgumentException("way must be at least 2 to have factors.");
        }
        else if (index >= this.factorsPerWay.length)
        {
            return null;
        }
        
        return this.factorsPerWay[way - 2];
    }
    
    /**
     * Gets the number of factors in the given way.
     * 
     * @param   way
     *      The way to get the factors for. 
     * @return 
     *      The number of factors for the given way (k_l), if there are any.
     *      For for ways less than 2, the number factors is 0.
     */
    public int getFactorCount(
        final int way)
    {
        if (way <= 1)
        {
            // No factors for this way.
            return 0;
        }

        
        final Matrix factors = this.getFactors(way);
        return (factors == null ? 0 : factors.getNumRows());
    }
    
    /**
     * Determines if the given way has factors in the machine. This means that
     * this way interactions are being modeled by it.
     * 
     * @param   way
     *      The way. Only ways of 2 or more can have factors.
     * @return 
     *      True if there are factors for that way. Otherwise, false.
     */
    public boolean hasFactors(
        final int way)
    {
        return this.getFactorCount(way) > 0;
    }
    
    /**
     * Sets the factors for a given way.
     * 
     * @param   way
     *      The way. Must be greater than 1 and less than the maximum number of
     *      ways for this factorization machine.
     * @param   factors 
     *      The k-by-d matrix of factors for the way, where k is the number of
     *      factors and d is the dimensionality of the input. Typically k is
     *      much less than d.
     */
    public void setFactors(
        final int way,
        final Matrix factors)
    {
        if (way < 2)
        {
            throw new IllegalArgumentException("way must be at least 2 to have factors.");
        }
        
        this.factorsPerWay[way - 2] = factors;
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
     * Gets the array of factors for each way of 2 and higher.
     * 
     * @return 
     *      The matrix of factors per way.
     */
    public Matrix[] getFactorsPerWay()
    {
        return this.factorsPerWay;
    }

    /**
     * Sets the array of factors for each way of 2 and higher. That means the
     * first element of the array is the set of factors for the 2-way 
     * interaction. The array itself cannot be null, but each individual entry
     * can be null.
     * 
     * @param   factorsPerWay 
     *      An array of k_l by d factor matrices for each way l starting from
     *      ways of 2. Cannot be null. Can be empty. Each factor
     *      matrix can be null to mean that interaction is not used.
     *      Currently, only up to 10-way interactions are supported, which 
     *      means the maximum size of the array is 8.
     */
    public void setFactorsPerWay(
        final Matrix... factorsPerWay)
    {
        ArgumentChecker.assertIsNotNull("factorsPerWay", factorsPerWay);
        final int maxNumWays = factorsPerWay.length + 1;
        if (maxNumWays >= PARTITIONS_PER_WAY.length)
        {
// TODO: Implement mechanism for supporting arbitrary numbers of ways by computing the partitions and coefficients.
            throw new IllegalArgumentException("Cannot currently have more than " + (PARTITIONS_PER_WAY.length - 1) + " way interactions.");
        }
        
        int expectedDimensionality = -1;
        for (final Matrix factors : factorsPerWay)
        {
            if (factors != null)
            {
                int d = factors.getNumColumns();
                if (expectedDimensionality < 0)
                {
                    expectedDimensionality = d;
                }
                else if (expectedDimensionality != d)
                {
                    throw new IllegalArgumentException(
                        "All factors matrices must have the same number of columns");
                }
            }
        }
        
        this.factorsPerWay = factorsPerWay;
    }
    
    /** The precomputed set of partitions for each of the numbers up to 10. 
     *  There must be a corresponding entry for the COEFFICIENTS_PER_WAY for
     *  each way. */
    protected static int[][][] PARTITIONS_PER_WAY =
    {
        { },
        { {1} },
        { {1, 1}, {2} },
        { {1, 1, 1}, {1, 2}, {3} },
        { {1, 1, 1, 1}, {1, 1, 2}, {1, 3}, {2, 2}, {4} },
        { {1, 1, 1, 1, 1}, {1, 1, 1, 2}, {1, 1, 3}, {1, 2, 2}, {1, 4}, {2, 3}, {5} },
        { {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 2}, {1, 1, 1, 3}, {1, 1, 2, 2}, {1, 1, 4}, {1, 2, 3}, {1, 5}, {2, 2, 2}, {2, 4}, {3, 3}, {6} },
        { {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 3}, {1, 1, 1, 2, 2}, {1, 1, 1, 4}, {1, 1, 2, 3}, {1, 1, 5}, {1, 2, 2, 2}, {1, 2, 4}, {1, 3, 3}, {1, 6}, {2, 2, 3}, {2, 5}, {3, 4}, {7} },
        { {1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 3}, {1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 4}, {1, 1, 1, 2, 3}, {1, 1, 1, 5}, {1, 1, 2, 2, 2}, {1, 1, 2, 4}, {1, 1, 3, 3}, {1, 1, 6}, {1, 2, 2, 3}, {1, 2, 5}, {1, 3, 4}, {1, 7}, {2, 2, 2, 2}, {2, 2, 4}, {2, 3, 3}, {2, 6}, {3, 5}, {4, 4}, {8} },
        { {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 3}, {1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 2, 3}, {1, 1, 1, 1, 5}, {1, 1, 1, 2, 2, 2}, {1, 1, 1, 2, 4}, {1, 1, 1, 3, 3}, {1, 1, 1, 6}, {1, 1, 2, 2, 3}, {1, 1, 2, 5}, {1, 1, 3, 4}, {1, 1, 7}, {1, 2, 2, 2, 2}, {1, 2, 2, 4}, {1, 2, 3, 3}, {1, 2, 6}, {1, 3, 5}, {1, 4, 4}, {1, 8}, {2, 2, 2, 3}, {2, 2, 5}, {2, 3, 4}, {2, 7}, {3, 3, 3}, {3, 6}, {4, 5}, {9} },
        { {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 3}, {1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 1, 2, 3}, {1, 1, 1, 1, 1, 5}, {1, 1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 2, 4}, {1, 1, 1, 1, 3, 3}, {1, 1, 1, 1, 6}, {1, 1, 1, 2, 2, 3}, {1, 1, 1, 2, 5}, {1, 1, 1, 3, 4}, {1, 1, 1, 7}, {1, 1, 2, 2, 2, 2}, {1, 1, 2, 2, 4}, {1, 1, 2, 3, 3}, {1, 1, 2, 6}, {1, 1, 3, 5}, {1, 1, 4, 4}, {1, 1, 8}, {1, 2, 2, 2, 3}, {1, 2, 2, 5}, {1, 2, 3, 4}, {1, 2, 7}, {1, 3, 3, 3}, {1, 3, 6}, {1, 4, 5}, {1, 9}, {2, 2, 2, 2, 2}, {2, 2, 2, 4}, {2, 2, 3, 3}, {2, 2, 6}, {2, 3, 5}, {2, 4, 4}, {2, 8}, {3, 3, 4}, {3, 7}, {4, 6}, {5, 5}, {10} }
    };
    
    /** The coefficients for the polynomials of each way, which correspond to
     *  the powers in each partition of the number n. It must be the same length
     *  as PARTITIONS_PER_WAY, which is 10. */
    protected static int[][] COEFFICIENTS_PER_WAY = 
    {
        {  },
        { 1 },
        { 1, -1 },
        { 1, -3, 2 },
        { 1, -6, 8, 3, -6 },
        { 1, -10, 20, 15, -30, -20, 24 },
        { 1, -15, 40, 45, -90, -120, 144, -15, 90, 40, -120 },
        { 1, -21, 70, 105, -210, -420, 504, -105, 630, 280, -840, 210, -504, -420, 720 },
        { 1, -28, 112, 210, -420, -1120, 1344, -420, 2520, 1120, -3360, 1680, -4032, -3360, 5760, 105, -1260, -1120, 3360, 2688, 1260, -5040 },
        { 1, -36, 168, 378, -756, -2520, 3024, -1260, 7560, 3360, -10080, 7560, -18144, -15120, 25920, 945, -11340, -10080, 30240, 24192, 11340, -45360, -2520, 9072, 15120, -25920, 2240, -20160, -18144, 40320 },
        { 1, -45, 240, 630, -1260, -5040, 6048, -3150, 18900, 8400, -25200, 25200, -60480, -50400, 86400, 4725, -56700, -50400, 151200, 120960, 56700, -226800, -25200, 90720, 151200, -259200, 22400, -201600, -181440, 403200, -945, 18900, 25200, -75600, -120960, -56700, 226800, -50400, 172800, 151200, 72576, -362880 }
    };
    
    /** The factorials for each way. It is the same length as 
     *  PARTITIONS_PER_WAY. That means that each entry i is i!. */
    protected static int[] FACTORIAL_PER_WAY = 
    {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800
    };
    
}
