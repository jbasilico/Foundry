/*
 * File:            AbstractLongMap.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.collection;

import gov.sandia.cognition.util.AbstractCloneableSerializable;
import gov.sandia.cognition.util.CloneableSerializable;
import java.util.AbstractMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * An abstract implementation of the {@link LongMap} interface.
 *
 * @param   <KeyType>
 *      The type of the key in the map.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public abstract class AbstractLongMap<KeyType>
    extends AbstractCloneableSerializable
    implements LongMap<KeyType>
{

    /**
     * Creates a new {@link AbstractLongMap}.
     */
    public AbstractLongMap()
    {
        super();
    }
    
    
    @Override
    public Map<KeyType, ? extends Number> asMap()
    {
        return new MapWrapper();
    }

    @Override
    public void setAll(
        final Iterable<? extends KeyType> keys,
        final long value)
    {
        for (KeyType key : keys)
        {
            this.set(key, value);
        }
    }

    @Override
    public long increment(
        final KeyType key)
    {
        return this.increment(key, 1);
    }

    @Override
    public long increment(
        final KeyType key,
        final long value)
    {
        final long newValue = this.get(key) + value;
        this.set(key, newValue);
        return newValue;
    }

    @Override
    public void incrementAll(
        final Iterable<? extends KeyType> keys)
    {
        for (KeyType key : keys)
        {
            this.increment(key);
        }
    }

    @Override
    public void incrementAll(
        final LongMap<? extends KeyType> other)
    {
        for (Entry<? extends KeyType> entry : other.entrySet())
        {
            this.increment(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public long decrement(
        final KeyType key)
    {
        return this.decrement(key, 1);
    }

    @Override
    public long decrement(
        final KeyType key,
        final long value)
    {
        return this.increment(key, -value);
    }

    @Override
    public void decrementAll(
        final Iterable<? extends KeyType> keys)
    {
        for (KeyType key : keys)
        {
            this.decrement(key);
        }
    }

    @Override
    public void decrementAll(
        final LongMap<? extends KeyType> other)
    {
        for (Entry<? extends KeyType> entry : other.entrySet())
        {
            this.decrement(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public long getMaxValue()
    {
        long maxValue = Long.MIN_VALUE;

        for (Entry<KeyType> entry : this.entrySet())
        {
            final long value = entry.getValue();
            if (value > maxValue)
            {
                maxValue = value;
            }
        }

        return maxValue;
    }

    @Override
    public long getMinValue()
    {
        long minValue = Long.MAX_VALUE;

        for (Entry<KeyType> entry : this.entrySet())
        {
            final long value = entry.getValue();
            if (value < minValue)
            {
                minValue = value;
            }
        }

        return minValue;
    }

    @Override
    public boolean isEmpty()
    {
        return this.size() == 0;
    }

    @Override
    public KeyType getMaxValueKey()
    {
        long maxValue = Long.MAX_VALUE;
        KeyType maxKey = null;

        for (Entry<KeyType> entry : this.entrySet())
        {
            final KeyType key = entry.getKey();
            final long value = entry.getValue();
            if (maxKey == null || value > maxValue)
            {
                maxKey = key;
                maxValue = value;
            }
        }

        return maxKey;
    }

    @Override
    public Set<KeyType> getMaxValueKeys()
    {
        long maxValue = Long.MIN_VALUE;
        final LinkedHashSet<KeyType> maxKeys = new LinkedHashSet<>();

        for (Entry<KeyType> entry : this.entrySet())
        {
            final KeyType key = entry.getKey();
            final long value = entry.getValue();
            if (value > maxValue)
            {
                maxKeys.clear();
                maxValue = value;
                maxKeys.add(key);
            }
            else if (value == maxValue)
            {
                maxKeys.add(key);
            }
        }

        return maxKeys;
    }

    @Override
    public KeyType getMinValueKey()
    {
        long minValue = Long.MAX_VALUE;
        KeyType minKey = null;

        for (Entry<KeyType> entry : this.entrySet())
        {
            final KeyType key = entry.getKey();
            final long value = entry.getValue();
            if (minKey == null || value < minValue)
            {
                minKey = key;
                minValue = value;
            }
        }

        return minKey;
    }

    @Override
    public Set<KeyType> getMinValueKeys()
    {
        long minValue = Long.MAX_VALUE;
        final LinkedHashSet<KeyType> minKeys = new LinkedHashSet<>();

        for (Entry<KeyType> entry : this.entrySet())
        {
            final KeyType key = entry.getKey();
            final long value = entry.getValue();
            if (value < minValue)
            {
                minKeys.clear();
                minValue = value;
                minKeys.add(key);
            }
            else if (value == minValue)
            {
                minKeys.add(key);
            }
        }

        return minKeys;
    }

    /**
     * Wrapper when using the asMap method
     */
    @SuppressWarnings("unchecked")
    protected class MapWrapper
        extends AbstractMap<KeyType, Long>
        implements CloneableSerializable
    {
        /**
         * Default constructor
         */
        protected MapWrapper()
        {
            super();
        }

        @Override
        public MapWrapper clone()
        {
            try
            {
                return (MapWrapper) super.clone();
            }
            catch (CloneNotSupportedException e)
            {
                throw new RuntimeException(e);
            }
        }

        @Override
        public int size()
        {
            return AbstractLongMap.this.size();
        }

        @Override
        public boolean isEmpty()
        {
            return AbstractLongMap.this.isEmpty();
        }

        @Override
        public boolean containsKey(
            final Object key)
        {
            return AbstractLongMap.this.containsKey((KeyType) key);
        }

        @Override
        public Long get(
            final Object key)
        {
            return AbstractLongMap.this.get((KeyType) key);
        }

        @Override
        public Long put(
            final KeyType key,
            final Long value)
        {
            if (value == null)
            {
                AbstractLongMap.this.set(key, 0);
            }
            else
            {
                AbstractLongMap.this.set(key, value);
            }
            return value;
        }

        @Override
        public Long remove(
            final Object key)
        {
            final long oldValue = AbstractLongMap.this.get((KeyType) key);
            AbstractLongMap.this.set((KeyType) key, 0);
            return oldValue;
        }

        @Override
        public void clear()
        {
            AbstractLongMap.this.clear();
        }

        @Override
        public Set<KeyType> keySet()
        {
            return AbstractLongMap.this.keySet();
        }

        @Override
        public Set<Map.Entry<KeyType, Long>> entrySet()
        {
            final LinkedHashSet<Map.Entry<KeyType, Long>> result =
                new LinkedHashSet<>(this.size());

            for (LongMap.Entry<KeyType> entry
                : AbstractLongMap.this.entrySet())
            {
                result.add(new AbstractMap.SimpleImmutableEntry<>(
                    entry.getKey(), entry.getValue()));
            }
            return result;
        }

    }
}
