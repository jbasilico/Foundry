/*
 * File:            AbstractDataCounter.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.statistics;

import gov.sandia.cognition.collection.AbstractMutableLongMap;
import gov.sandia.cognition.collection.LongMap;
import gov.sandia.cognition.collection.ScalarMap;
import gov.sandia.cognition.math.MathUtil;
import gov.sandia.cognition.math.MutableLong;
import gov.sandia.cognition.math.matrix.DefaultInfiniteVector;
import gov.sandia.cognition.math.matrix.InfiniteVector;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * An abstract implementation of the {@code DataCounter} interface.
 * 
 * @param   <KeyType>
 *      The type of the key in the map.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public abstract class AbstractCountDistribution<KeyType>
    extends AbstractMutableLongMap<KeyType>
    implements CountDistribution<KeyType>
{

    /**
     * Creates a new {@code AbstractDataCounter}.
     * 
     * @param map
     *      Map that stores the data.
     */
    public AbstractCountDistribution(
        final Map<KeyType, MutableLong> map)
    {
        super(map);
    }

    @Override
    @SuppressWarnings("unchecked")
    public AbstractCountDistribution<KeyType> clone()
    {
        return (AbstractCountDistribution<KeyType>) super.clone();
    }

    @Override
    public int getDomainSize()
    {
        return this.map.size();
    }

    @Override
    public double getEntropy()
    {
        double entropy = 0.0;
        final long total = this.getTotal();
        final double denom = (total != 0.0) ? total : 1.0;
        for (LongMap.Entry<KeyType> entry : this.entrySet())
        {
            double p = entry.getValue() / denom;
            if (p != 0.0)
            {
                entropy -= p * MathUtil.log2(p);
            }
        }
        return entropy;
    }

    @Override
    public double getLogFraction(
        final KeyType key)
    {
        final long total = this.getTotal();
        return (total != 0) ? (Math.log(this.get(key)) - Math.log(total))
            : Double.NEGATIVE_INFINITY;
    }

    @Override
    public double getFraction(
        final KeyType key)
    {
        final long total = this.getTotal();
        return (total != 0) ? (this.get(key) / this.getTotal()) : 0.0;
    }

    @Override
    public KeyType sample(
        final Random random)
    {
        double w = random.nextDouble() * this.getTotal();
        for (LongMap.Entry<KeyType> entry : this.entrySet())
        {
            w -= entry.getValue();
            if (w <= 0.0)
            {
                return entry.getKey();
            }
        }
        return null;
    }

    @Override
    public ArrayList<KeyType> sample(
        final Random random,
        final int numSamples)
    {
        final ArrayList<KeyType> result = new ArrayList<KeyType>(numSamples);
        this.sampleInto(random, numSamples, result);
        return result;
    }

    @Override
    public void sampleInto(
        final Random random,
        final int sampleCount,
        final Collection<? super KeyType> output)
    {
        // Compute the cumulative weights
        final int size = this.getDomainSize();
        double[] cumulativeWeights = new double[size];
        double cumulativeSum = 0.0;
        ArrayList<KeyType> domain = new ArrayList<KeyType>(size);
        int index = 0;
        for (LongMap.Entry<KeyType> entry : this.entrySet())
        {
            domain.add(entry.getKey());
            final double value = entry.getValue();
            cumulativeSum += value;
            cumulativeWeights[index] = cumulativeSum;
            index++;
        }
        
        ProbabilityMassFunctionUtil.sampleMultipleInto(
            cumulativeWeights, domain, random, sampleCount, output);
    }

    @Override
    public InfiniteVector<KeyType> toInfiniteVector()
    {
        final DefaultInfiniteVector<KeyType> result =
            new DefaultInfiniteVector<KeyType>(this.size());

        for (LongMap.Entry<KeyType> entry : this.entrySet())
        {
            result.set(entry.getKey(), entry.getValue());
        }

        return result;
    }

    @Override
    public void fromInfiniteVector(
        final InfiniteVector<? extends KeyType> vector)
    {
        this.clear();

        for (ScalarMap.Entry<? extends KeyType> entry : vector.entrySet())
        {
            this.set(entry.getKey(), (long) entry.getValue());
        }
    }

    @Override
    public long getMaxValue()
    {
        if (this.getTotal() <= 0)
        {
            return 0;
        }
        else
        {
            return super.getMaxValue();
        }
    }

    @Override
    public long getMinValue()
    {
        if (this.getTotal() <= 0)
        {
            return 0;
        }
        else
        {
            return super.getMinValue();
        }
    }

    @Override
    public Set<KeyType> getDomain()
    {
        return this.keySet();
    }

}
