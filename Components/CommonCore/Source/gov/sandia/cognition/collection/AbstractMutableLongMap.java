/*
 * File:            AbstractMutableLongMap.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.collection;

import gov.sandia.cognition.math.MutableLong;
import java.util.AbstractSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;


/**
 * A partial implementation of the {@link LongMap} interface using a
 * {@link MutableLong} value.
 *
 * @param   <KeyType>
 *      The type of the key in the map.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public abstract class AbstractMutableLongMap<KeyType>
    extends AbstractLongMap<KeyType>
{

    /**
     * Map backing that performs the storage.
     */
    protected Map<KeyType, MutableLong> map;

    /** 
     * Creates a new instance of {@link AbstractMutableLongMap}.
     * 
     * @param   map
     *      The backing map that the data is stored in.
     */
    public AbstractMutableLongMap(
        final Map<KeyType, MutableLong> map)
    {
        super();
        
        this.map = map;
    }

    @Override
    public AbstractMutableLongMap<KeyType> clone()
    {
        @SuppressWarnings("unchecked")
        final AbstractMutableLongMap<KeyType> clone =
            (AbstractMutableLongMap<KeyType>) super.clone();
        // NOTE: there is the potential for a problem if this.map isn't a
        // LinkedHashMap!
        clone.map = new LinkedHashMap<>(this.size());
        clone.incrementAll(this);
        return clone;
    }

    @Override
    public Map<KeyType, MutableLong> asMap()
    {
        return this.map;
    }

    /**
     * Removes entries from the map with value of 0.
     */
    public void compact()
    {
        // We can't use entrySet to remove null/zero elements because it throws
        // a ConcurrentModificationException
        // We can't use the keySet either because that throws an Exception,
        // so we need to clone it first
        final LinkedList<KeyType> removeKeys = new LinkedList<>();
        for (final Map.Entry<KeyType, MutableLong> entry : this.map.entrySet())
        {
            final MutableLong value = entry.getValue();
            if (value.value == 0)
            {
                removeKeys.add(entry.getKey());
            }
        }
        for (KeyType key : removeKeys)
        {
            this.map.remove(key);
        }
    }

    @Override
    public long get(
        final KeyType key)
    {
        final MutableLong entry = this.map.get(key);
        if (entry == null)
        {
            return 0;
        }
        else
        {
            return entry.value;
        }
    }

    @Override
    public void set(
        final KeyType key,
        final long value)
    {
        final MutableLong entry = this.map.get(key);
        if (entry == null)
        {
            // Only need to allocate if it's not null
            if (value != 0)
            {
                this.map.put(key, new MutableLong(value));
            }
        }
        // I've commented this out, because I think there's the potential for
        // memory thrashing if this is left in... call compact() method instead?
        //  -- krdixon, 2011-06-27
//        else if( value == 0.0 )
//        {
//            this.map.remove(key);
//        }
        else
        {
            entry.value = value;
        }
    }

    @Override
    public long increment(
        final KeyType key,
        final long value)
    {
        final MutableLong entry = this.map.get(key);
        long newValue;
        if (entry == null)
        {
            if (value != 0.0)
            {
                // It's best to avoid this.set() here as it could mess up
                // our total tracker in some subclasses...
                // Also it's more efficient this way (avoid another get)
                this.map.put(key, new MutableLong(value));
            }
            newValue = value;
        }
        else
        {
            entry.value += value;
            newValue = entry.value;
        }
        return newValue;
    }

    @Override
    public void clear()
    {
        this.map.clear();
    }

    @Override
    public SimpleEntrySet<KeyType> entrySet()
    {
        return new SimpleEntrySet<KeyType>(this.map);
    }

    @Override
    public Set<KeyType> keySet()
    {
        return this.map.keySet();
    }

    @Override
    public boolean containsKey(
        final KeyType key)
    {
        return this.map.containsKey(key);
    }

    @Override
    public int size()
    {
        return this.map.size();
    }

    /**
     * Simple Entry Set for DefaultInfiniteVector
     * @param <KeyType>
     *      The type of the key in the map.
     */
    protected static class SimpleEntrySet<KeyType>
        extends AbstractSet<SimpleEntry<KeyType>>
    {

        /** The map of keys to mutable long values. */
        protected Map<KeyType, MutableLong> map;

        /**
         * Creates a new instance of {@link SimpleEntrySet}.
         * 
         * @param   map
         *      The backing map.
         */
        public SimpleEntrySet(
            final Map<KeyType, MutableLong> map)
        {
            super();
            
            this.map = map;
        }

        @Override
        public Iterator<AbstractMutableLongMap.SimpleEntry<KeyType>> iterator()
        {
            return new SimpleIterator<>(map.entrySet().iterator());
        }

        @Override
        public int size()
        {
            return map.size();
        }

    }

    /**
     * Simple iterator for {@link AbstractMutableLongMap}.
     * 
     * @param <KeyType>
     *      The type of the key in the map.
     */
    protected static class SimpleIterator<KeyType>
        extends Object
        implements Iterator<AbstractMutableLongMap.SimpleEntry<KeyType>>
    {

        /**
         * Iterator that does all the work.
         */
        private Iterator<? extends Map.Entry<KeyType, MutableLong>> delegate;

        /**
         * Creates a new {@link SimpleIterator}.
         * 
         * @param delegate
         *      Iterator that does all the work.
         */
        public SimpleIterator(
            final Iterator<? extends Map.Entry<KeyType, MutableLong>> delegate)
        {
            super();
            
            this.delegate = delegate;
        }

        @Override
        public boolean hasNext()
        {
            return this.delegate.hasNext();
        }

        @Override
        public AbstractMutableLongMap.SimpleEntry<KeyType> next()
        {
            final Map.Entry<KeyType,MutableLong> entry = this.delegate.next();
            return new AbstractMutableLongMap.SimpleEntry<>(
                entry.getKey(), entry.getValue());
        }

        @Override
        public void remove()
        {
            this.delegate.remove();
        }

    }


    /**
     * Entry for the {@link AbstractLongMap}.
     * 
     * @param <KeyType>
     *      The type of the key in the map.
     */
    protected static class SimpleEntry<KeyType>
        extends Object
        implements LongMap.Entry<KeyType>
    {

        /** Key associated with this entry. */
        protected KeyType key;

        /** Value associated with the entry. */
        protected MutableLong value;

        /**
         * Creates a new instance of {@link SimpleEntry}.
         * 
         * @param key
         *      Key associated with the Entry.
         * @param value
         *      Value associated with the Entry.
         */
        public SimpleEntry(
            final KeyType key,
            final MutableLong value)
        {
            super();
            
            this.key = key;
            this.value = value;
        }

        @Override
        public KeyType getKey()
        {
            return this.key;
        }

        @Override
        public long getValue()
        {
            return value.value;
        }

        @Override
        public void setValue(
            long value)
        {
            this.value.value = value;
        }

    }

}
