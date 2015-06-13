/*
 * File:            CountDistribution.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.statistics;

import gov.sandia.cognition.collection.LongMap;
import gov.sandia.cognition.math.matrix.InfiniteVector;

/**
 * An abstract implementation of the {@code CountDistribution} interface.
 * 
 * @param   <DataType>
 *      The type of the key in the map.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public interface CountDistribution<DataType>
    extends DiscreteDistribution<DataType>,
        EstimableDistribution<DataType, CountDistribution<DataType>>,
        LongMap<DataType>
{

    @Override
    public CountDistribution<DataType> clone();

    /**
     * Converts this data distribution to an infinite vector.
     *
     * @return
     *      A new {@code InfiniteVector} with values from this data
     *      distribution.
     */
    public InfiniteVector<DataType> toInfiniteVector();

    /**
     * Replaces the entries in this data distribution with the entries in the
     * given infinite vector.
     *
     * @param   vector
     *      The infinite vector to use to populate this data distribution.
     */
    public void fromInfiniteVector(
        final InfiniteVector<? extends DataType> vector);

    /**
     * Computes the information-theoretic entropy of the vector in bits.
     *
     * @return
     *      Entropy in bits of the distribution.
     */
    public double getEntropy();

    /**
     * Gets the fraction of the counts represented by the given key.
     *
     * @param   key
     *      The key.
     * @return
     *      The fraction of the total count represented by the key, if it
     *      exists. Otherwise, 0.0.
     */
    public double getFraction(
        final DataType key);

    /**
     * Gets the natural logarithm of the fraction of the counts represented
     * by the given key.
     *
     * @param key
     * Key to consider
     * @return
     * Natural logarithm of the fraction of the counts represented by the key
     */
    public double getLogFraction(
        final DataType key);

    /**
     * Gets the total (sum) of the values in the distribution.
     *
     * @return
     *      The sum of the values in the distribution.
     */
    public long getTotal();

    @Override
    public DistributionEstimator<DataType, ? extends CountDistribution<DataType>> getEstimator();

    @Override
    public CountDistribution.PMF<DataType> getProbabilityFunction();

    /**
     * Interface for the probability mass function (PMF) of a data distribution.
     *
     * @param <KeyType>
     *      Type of data stored at the indices, the hash keys.
     */
    public static interface PMF<KeyType>
        extends CountDistribution<KeyType>,
        ProbabilityMassFunction<KeyType>
    {
    }

}
