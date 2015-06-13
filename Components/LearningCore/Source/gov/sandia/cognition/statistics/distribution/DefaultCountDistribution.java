/*
 * File:            DefaultCountDistribution.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.statistics.distribution;

import gov.sandia.cognition.factory.Factory;
import gov.sandia.cognition.learning.algorithm.AbstractBatchAndIncrementalLearner;
import gov.sandia.cognition.math.MutableLong;
import gov.sandia.cognition.statistics.AbstractCountDistribution;
import gov.sandia.cognition.statistics.CountDistribution;
import gov.sandia.cognition.statistics.DistributionEstimator;
import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.ArgumentChecker;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A default implementation of {@code CountDistribution} that uses a
 * backing map.
 * 
 * @param   <KeyType>
 *      The type of the key in the distribution.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public class DefaultCountDistribution<KeyType>
    extends AbstractCountDistribution<KeyType>
{
    
    /** Default initial capacity is {@value}. */
    public static final int DEFAULT_INITIAL_CAPACITY = 10;
    
    /** The total of the counts in the distribution. */
    protected long total;
    
    /**
     * Creates a new {@link DefaultCountDistribution}.
     */
    public DefaultCountDistribution()
    {
        this(DEFAULT_INITIAL_CAPACITY);
    }

    /**
     * Creates a new {@link DefaultCountDistribution} with the given capacity.
     * 
     * @param initialCapacity
     *      Initial capacity of the backing map.
     */
    public DefaultCountDistribution(
        final int initialCapacity)
    {
        this(new LinkedHashMap<KeyType, MutableLong>(initialCapacity), 0);
    }

    /**
     * Creates a new {@link DefaultCountDistribution} that is a copy of the 
     * given count distribution.
     * 
     * @param other
     *      The other distribution to copy.
     */
    public DefaultCountDistribution(
        final CountDistribution<? extends KeyType> other)
    {
        this(new LinkedHashMap<KeyType, MutableLong>(other.size()), 0);
        
        this.incrementAll(other);
    }

    /**
     * Creates a new {@link DefaultCountDistribution} that is initialized by
     * counting the elements in the given iterable.
     * 
     * @param data
     *      Iterable of keys to count.
     */
    public DefaultCountDistribution(
        final Iterable<? extends KeyType> data)
    {
        this();
        
        this.incrementAll(data);
    }

    /**
     * Creates a new {@link DefaultCountDistribution} with the given backing
     * map and total.
     * 
     * @param map
     *      The backing map that stores the data.
     * @param total
     *      The sum of all values in the map.
     */
    protected DefaultCountDistribution(
        final Map<KeyType, MutableLong> map,
        final long total)
    {
        super(map);
        
        this.total = total;
    }

    @Override
    public DefaultCountDistribution<KeyType> clone()
    {
        final DefaultCountDistribution<KeyType> clone =
            (DefaultCountDistribution<KeyType>) super.clone();
        
        // We have to manually reset "total" because super.super.clone
        // calls "incrementAll", which will, in turn, increment the total
        // So we'd end up with twice the total.
        clone.total = this.total;
        return clone;
    }

    @Override
    public long increment(
        final KeyType key,
        final long value)
    {
        final MutableLong entry = this.map.get(key);
        long newValue = 0;
        long delta;
        if (entry == null)
        {
            if (value > 0)
            {
                // It's best to avoid this.set() here as it could mess up
                // our total tracker in some subclasses...
                // Also it's more efficient this way (avoid another get)
                this.map.put(key, new MutableLong(value));
                delta = value;
                newValue = value;
            }
            else
            {
                delta = 0;
            }
        }
        else
        {
            if ((entry.value + value) >= 0)
            {
                delta = value;
                entry.value += value;
                newValue = entry.value;
            }
            else
            {
                delta = -entry.value;
                entry.value = 0;
            }
            
        }

        this.total += delta;
        return newValue;
    }

    @Override
    public void set(
        final KeyType key,
        final long value)
    {
        // I decided not to call super.set because it would result in me
        // having to perform an extra call to this.map.get
        final MutableLong entry = this.map.get(key);
        if (entry == null)
        {
            // Only need to allocate if it's not null
            if (value > 0)
            {
                this.map.put(key, new MutableLong(value));
                this.total += value;
            }
        }
        else if (value > 0)
        {
            this.total += value - entry.value;
            entry.value = value;
        }
        else
        {
            this.total -= entry.value;
            entry.value = 0;
        }
    }

    @Override
    public long getTotal()
    {
        return this.total;
    }
    
    @Override
    public void clear()
    {
        super.clear();
        this.total = 0;
    }

    @Override
    public DistributionEstimator<KeyType, ? extends CountDistribution<KeyType>> getEstimator()
    {
        return new DefaultCountDistribution.Estimator<>();
    }

    @Override
    public CountDistribution.PMF<KeyType> getProbabilityFunction()
    {
        return new DefaultCountDistribution.PMF<>(this);
    }

    /**
     * Gets the average value of all keys in the distribution, that is, the
     * total value divided by the number of keys (even zero-value keys).
     * 
     * @return
     *      Average value of all keys in the distribution.
     */
    public double getMeanValue()
    {
        final int domainSize = this.getDomainSize();
        
        if (domainSize <= 0)
        {
            return 0.0;
        }
        else
        {
            return (double) this.getTotal() / domainSize;
        }
    }

    /**
     * PMF of the {@link DefaultCountDistribution}.
     * 
     * @param <KeyType>
     *      Type of Key in the distribution
     */
    public static class PMF<KeyType>
        extends DefaultCountDistribution<KeyType>
        implements CountDistribution.PMF<KeyType>
    {

        /**
         * Default constructor.
         */
        public PMF()
        {
            super();
        }

        /**
         * Copy constructor.
         * 
         * @param other
         *      CountDistribution to copy
         */
        public PMF(
            final CountDistribution<KeyType> other)
        {
            super(other);
        }

        /**
         * Creates a new instance of DefaultCountDistribution.PMF.
         * 
         * @param initialCapacity
         *      Initial capacity of the backing map.
         */
        public PMF(
            final int initialCapacity)
        {
            super(initialCapacity);
        }

        /**
         * Creates a new instance of DefaultCountDistribution.PMF.
         *
         * @param data 
         *      Data to create the distribution.
         */
        public PMF(
            final Iterable<? extends KeyType> data)
        {
            super(data);
        }

        @Override
        public double logEvaluate(
            KeyType input)
        {
            return this.getLogFraction(input);
        }

        @Override
        public Double evaluate(
            KeyType input)
        {
            return this.getFraction(input);
        }

        @Override
        public DefaultCountDistribution.PMF<KeyType> getProbabilityFunction()
        {
            return this;
        }

    }

    /**
     * Estimator for a {@link DefaultCountDistribution}.
     * 
     * @param <KeyType>
     *      Type of Key in the distribution.
     */
    public static class Estimator<KeyType>
        extends AbstractBatchAndIncrementalLearner<KeyType, DefaultCountDistribution.PMF<KeyType>>
        implements DistributionEstimator<KeyType, DefaultCountDistribution.PMF<KeyType>>
    {

        /**
         * Default constructor.
         */
        public Estimator()
        {
            super();
        }

        @Override
        public DefaultCountDistribution.PMF<KeyType> createInitialLearnedObject()
        {
            return new DefaultCountDistribution.PMF<KeyType>();
        }

        @Override
        public void update(
            final DefaultCountDistribution.PMF<KeyType> target,
            final KeyType data)
        {
            target.increment(data, 1);
        }

    }

    /**
     * A factory for {@code DefaultCountDistribution} objects using some given
     * initial capacity for them.
     *
     * @param   <DataType>
     *      The type of data for the factory.
     */
    public static class DefaultFactory<DataType>
        extends AbstractCloneableSerializable
        implements Factory<DefaultCountDistribution<DataType>>
    {

        /** The initial domain capacity. */
        protected int initialDomainCapacity;

        /**
         * Creates a new {@code DefaultFactory} with a default
         * initial domain capacity.
         */
        public DefaultFactory()
        {
            this(DEFAULT_INITIAL_CAPACITY);
        }

        /**
         * Creates a new {@code DefaultFactory} with a given
         * initial domain capacity.
         *
         * @param   initialDomainCapacity
         *      The initial capacity for the domain. Must be positive.
         */
        public DefaultFactory(
            final int initialDomainCapacity)
        {
            super();

            this.setInitialDomainCapacity(initialDomainCapacity);
        }

        @Override
        public DefaultCountDistribution<DataType> create()
        {
            // Create the histogram.
            return new DefaultCountDistribution<>(
                this.getInitialDomainCapacity());
        }

        /**
         * Gets the initial domain capacity.
         *
         * @return
         *      The initial domain capacity. Must be positive.
         */
        public int getInitialDomainCapacity()
        {
            return this.initialDomainCapacity;
        }

        /**
         * Sets the initial domain capacity.
         *
         * @param   initialDomainCapacity
         *      The initial domain capacity. Must be positive.
         */
        public void setInitialDomainCapacity(
            final int initialDomainCapacity)
        {
            ArgumentChecker.assertIsPositive("initialDomainCapacity",
                initialDomainCapacity);
            this.initialDomainCapacity = initialDomainCapacity;
        }

    }

}
