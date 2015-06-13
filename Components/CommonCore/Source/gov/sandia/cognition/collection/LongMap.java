/*
 * File:            LongMap.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.collection;

import java.util.Map;
import java.util.Set;

/**
 * An interface for a mapping of objects to integer values represented as
 * longs.
 *
 * @param   <KeyType>
 *      The type of the key in the map.
 * @author  Justin Basiico
 * @since   3.4.0
 */
public interface LongMap<KeyType>
    extends NumericMap<KeyType>
{
    
    /**
     * Gets a {@code java.util.Map} that contains the same data as in this
     * scalar map.
     *
     * @return
     *      The {@code Map} version of this data structure.
     */
    public Map<KeyType, ? extends Number> asMap();

    /**
     * Gets the value associated with a given key. If the key does not exist,
     * then 0 is returned.
     *
     * @param   key
     *      A key.
     * @return
     *      The value associated with the key or 0 if it does not exist.
     */
    public long get(
        final KeyType key);

    /**
     * Sets the value associated with a given key. In some cases if the value is
     * 0, the key may be removed from the map.
     *
     * @param   key
     *      A key.
     * @param   value
     *      The value to associate with the key.
     */
    public void set(
        final KeyType key,
        final long value);

    /**
     * Sets all the given keys to the given value.
     *
     * @param   keys
     *      A list of keys.
     * @param   value
     *      The value to associate with all the given keys.
     */
    public void setAll(
        final Iterable<? extends KeyType> keys,
        final long value);

    /**
     * Increments the value associated with the given key by 1.
     *
     * @param   key
     *      A key.
     * @return
     *      The new value associated with the key.
     */
    public long increment(
        final KeyType key);

    /**
     * Increments the value associated with the given key by the given amount.
     *
     * @param   key
     *      A key.
     * @param value
     *      The amount to increment the value associated with the given key by.
     * @return
     *      The new value associated with the key.
     */
    public long increment(
        final KeyType key,
        final long value);

    /**
     * Increments the values associated all of the given keys by 1.
     *
     * @param   keys
     *      A list of keys.
     */
    public void incrementAll(
        final Iterable<? extends KeyType> keys);

    /**
     * Increments all the keys in this map by the values in the other one.
     *
     * @param   other
     *      The other map.
     */
    public void incrementAll(
        final LongMap<? extends KeyType> other);

    /**
     * Decrements the value associated with a given key by 1.
     *
     * @param   key
     *      A key.
     * @return
     *      The new value associated with the key.
     */
    public long decrement(
        final KeyType key);

    /**
     * Decrements the value associated with the given key by the given amount.
     *
     * @param   key
     *      A key.
     * @param value
     *      The amount to decrement the value associated with the given key by.
     * @return
     *      The new value associated with the key.
     */
    public long decrement(
        final KeyType key,
        final long value);

    /**
     * Decrements the values associated all of the given keys by 1.
     *
     * @param   keys
     *      A list of keys.
     */
    public void decrementAll(
        final Iterable<? extends KeyType> keys);

    /**
     * Decrements all the keys in this map by the values in the other one.
     *
     * @param   other
     *      The other map.
     */
    public void decrementAll(
        final LongMap<? extends KeyType> other);

    /**
     * The maximum value associated with any key in the map.
     *
     * @return
     *      The maximum value associated with any key in the map. If the map
     *      is empty, then Double.NEGATIVE_INFINITY is returned.
     */
    public long getMaxValue();

    /**
     * The minimum value associated with any key in the map.
     *
     * @return
     *      The minimum value associated with any key in the map. If the map
     *      is empty, then Double.POSITIVE_INFINITY is returned.
     */
    public long getMinValue();

    /**
     * Gets the set of entries in this scalar map.
     *
     * @return
     *      The set of entries in the scalar map.
     */
    public Set<? extends LongMap.Entry<KeyType>> entrySet();

    /**
     * An entry in a scalar map. Similar to a Map.Entry.
     *
     * @param   <KeyType>
     *      The type of key in the map.
     * @see     java.util.Map.Entry
     */
    public static interface Entry<KeyType>
    {
        
        /**
         * Gets the key.
         *
         * @return
         *      The key.
         */
        public KeyType getKey();

        /**
         * Gets the value associated with the key.
         *
         * @return
         *      The value.
         */
        public long getValue();

        /**
         * Sets the value associated with the key. Optional operation.
         *
         * @param   value
         *      The value.
         */
        public void setValue(
            final long value);
    }
}
