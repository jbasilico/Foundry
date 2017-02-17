/*
 * File:            DefaultCountDistributionTest.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.statistics.distribution;

import gov.sandia.cognition.math.MutableLong;
import gov.sandia.cognition.math.matrix.DefaultInfiniteVector;
import gov.sandia.cognition.math.matrix.InfiniteVector;
import gov.sandia.cognition.statistics.CountDistribution;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for class {@link DefaultCountDistribution}.
 * 
 * @author  Justin Basiico
 * @since   3.4.4
 */
public class DefaultCountDistributionTest
    extends Object
{
    
    protected Random random = new Random(130031);
    
    /**
     * Creates a new test.
     */
    public DefaultCountDistributionTest()
    {
        super();
    }

    public DefaultCountDistribution<String> createEmpty()
    {
        return new DefaultCountDistribution<>();
    }
    
    public DefaultCountDistribution<String> createExample()
    {
        return new DefaultCountDistribution<>(
            Arrays.asList("a", "b", "a", "c"));
    }
    /**
     * Test of constructors of class DefaultCountDistribution.
     */
    @Test
    public void testConstructors()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.isEmpty());
        
        instance = new DefaultCountDistribution<>(11);
        assertTrue(instance.isEmpty());
        
        instance = new DefaultCountDistribution<>(
            Arrays.asList("a", "b", "a", "c"));
        assertEquals(2, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(1, instance.get("c"));
        assertEquals(4, instance.getTotal());
        
        instance = new DefaultCountDistribution<>(instance);
        assertEquals(2, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(1, instance.get("c"));
        assertEquals(4, instance.getTotal());
    }
    /**
     * Test of clone method, of class DefaultCountDistribution.
     */
    @Test
    public void testClone()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        DefaultCountDistribution<String> clone = instance.clone();
        assertNotNull(clone);
        assertNotSame(instance, clone);
        assertNotSame(clone, instance.clone());
        
        assertTrue(instance.isEmpty());
        assertTrue(clone.isEmpty());
        clone.increment("a");
        assertTrue(instance.isEmpty());
        assertEquals(1, clone.getTotal());
        
    }

    /**
     * Test of increment method, of class DefaultCountDistribution.
     */
    @Test
    public void testIncrement()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0, instance.getTotal());
        
        assertEquals(1, instance.increment("a"));
        assertEquals(2, instance.increment("a"));
        assertEquals(7, instance.increment("a", 5));
        assertEquals(7, instance.get("a"));
        assertEquals(7, instance.getTotal());
        assertEquals(1, instance.increment("b"));
        assertEquals(1, instance.get("b"));
        assertEquals(8, instance.getTotal());
        assertEquals(1, instance.increment("c", 1));
        assertEquals(4, instance.increment("c", 3));
        assertEquals(4, instance.get("c"));
        assertEquals(12, instance.getTotal());
        assertEquals(2, instance.increment("c", -2));
        assertEquals(2, instance.get("c"));
        assertEquals(10, instance.getTotal());
        assertEquals(2, instance.increment("c", 0));
        assertEquals(2, instance.get("c"));
        assertEquals(10, instance.getTotal());
        
        assertEquals(0, instance.increment("d", -2));
        assertEquals(0, instance.get("d"));
        assertEquals(10, instance.getTotal());
        
        assertEquals(3, instance.getDomain().size());
    }
    
    /**
     * Test of decrement method, of class DefaultCountDistribution.
     */
    @Test
    public void testDecrement()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0, instance.getTotal());
        
        assertEquals(1, instance.increment("a"));
        assertEquals(2, instance.increment("a"));
        assertEquals(7, instance.increment("a", 5));
        assertEquals(7, instance.get("a"));
        assertEquals(7, instance.getTotal());
        assertEquals(1, instance.increment("b"));
        assertEquals(1, instance.get("b"));
        assertEquals(8, instance.getTotal());
        assertEquals(1, instance.increment("c", 1));
        assertEquals(4, instance.decrement("c", -3));
        assertEquals(4, instance.get("c"));
        assertEquals(12, instance.getTotal());
        assertEquals(2, instance.decrement("c", 2));
        assertEquals(2, instance.get("c"));
        assertEquals(10, instance.getTotal());
        assertEquals(2, instance.decrement("c", 0));
        assertEquals(2, instance.get("c"));
        assertEquals(10, instance.getTotal());
        
        assertEquals(0, instance.decrement("d", 2));
        assertEquals(0, instance.get("d"));
        assertEquals(10, instance.getTotal());
        
        assertEquals(3, instance.getDomain().size());
    }
    
    
    /**
     * Test of asMap method, of class DefaultCountDistribution.
     */
    @Test
    public void testAsMap()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        assertTrue(instance.asMap().isEmpty());

        instance = this.createExample();
        Map<String, MutableLong> result = instance.asMap();
        assertEquals(3, result.size());
        assertEquals(2, result.get("a").value);
        assertEquals(1, result.get("b").value);
        assertEquals(1, result.get("c").value);
    }
    
    /**
     * Test of entrySet method, of class DefaultCountDistribution.
     */
    @Test
    public void testEntrySet()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        assertTrue(instance.entrySet().isEmpty());
        
        instance = this.createExample();
        assertEquals(3, instance.entrySet().size());
    }

    /**
     * Test of keySet method, of class DefaultCountDistribution.
     */
    @Test
    public void testKeySet()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        assertTrue(instance.keySet().isEmpty());

        instance = this.createExample();
        Set<String> result = instance.keySet();
        assertEquals(3, result.size());
        assertTrue(result.containsAll(Arrays.asList("a", "b", "c")));
    }

    /**
     * Test of containsKey method, of class DefaultCountDistribution.
     */
    @Test
    public void testContainsKey()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        assertFalse(instance.containsKey("a"));
        assertFalse(instance.containsKey("b"));
        assertFalse(instance.containsKey("c"));
        assertFalse(instance.containsKey("d"));
        
        instance = this.createExample();
        assertTrue(instance.containsKey("a"));
        assertTrue(instance.containsKey("b"));
        assertTrue(instance.containsKey("c"));
        assertFalse(instance.containsKey("d"));
    }

    /**
     * Test of size method, of class DefaultCountDistribution.
     */
    @Test
    public void testSize()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        assertEquals(0, instance.size());
        
        instance = this.createExample();
        assertEquals(3, instance.size());
    }
    
    /**
     * Test of compact method, of class DefaultCountDistribution.
     */
    @Test
    public void testCompact()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        instance.compact();
        assertTrue(instance.isEmpty());
        
        instance.increment("a", 3);
        instance.increment("b");
        instance.compact();
        assertEquals(2, instance.getDomainSize());
        assertEquals(4, instance.getTotal());
        assertEquals(3, instance.get("a"));
        assertEquals(1, instance.get("b"));
        
        instance.decrement("a");
        instance.decrement("b");
        assertEquals(2, instance.getDomainSize());
        assertEquals(2, instance.getTotal());
        assertEquals(2, instance.get("a"));
        assertEquals(0, instance.get("b"));
        
        instance.compact();
        assertEquals(1, instance.getDomainSize());
        assertEquals(2, instance.getTotal());
        assertEquals(2, instance.get("a"));
        assertEquals(0, instance.get("b"));
    }

    /**
     * Test of setAll method, of class DefaultCountDistribution.
     */
    @Test
    public void testSetAll()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        instance.setAll(Arrays.asList("a", "b", "c"), 5);
        assertEquals(15, instance.getTotal());
        assertEquals(5, instance.get("a"));
        assertEquals(5, instance.get("b"));
        assertEquals(5, instance.get("c"));
        assertEquals(0, instance.get("d"));
    }

    /**
     * Test of incrementAll method, of class DefaultCountDistribution.
     */
    @Test
    public void testIncrementAll()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        instance.incrementAll(Arrays.asList());
        assertTrue(instance.isEmpty());
        
        instance.incrementAll(Arrays.asList("a", "b", "a", "c"));
        assertEquals(3, instance.getDomainSize());
        assertEquals(4, instance.getTotal());
        assertEquals(2, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(1, instance.get("c"));
        
        instance.incrementAll(new DefaultCountDistribution<>(Arrays.asList("a", "d", "c")));
        assertEquals(4, instance.getDomainSize());
        assertEquals(7, instance.getTotal());
        assertEquals(3, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(2, instance.get("c"));
        assertEquals(1, instance.get("d"));
    }


    /**
     * Test of decrementAll method, of class DefaultCountDistribution.
     */
    @Test
    public void testDecrementAll()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        instance.decrementAll(Arrays.asList());
        assertTrue(instance.isEmpty());
        
        instance.incrementAll(Arrays.asList("a", "b", "a", "c", "d"));
        instance.decrementAll(Arrays.asList("a", "c"));
        assertEquals(3, instance.getTotal());
        assertEquals(1, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(1, instance.get("d"));
        
        instance.decrementAll(new DefaultCountDistribution<>(Arrays.asList("d", "a")));
        assertEquals(1, instance.getTotal());
        assertEquals(0, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(0, instance.get("c"));   
        assertEquals(0, instance.get("d"));     
    }
    
    /**
     * Test of set method, of class DefaultCountDistribution.
     */
    @Test
    public void testSet()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0, instance.getTotal());
        
        instance.set("a", 7);
        assertEquals(7, instance.get("a"));
        assertEquals(7, instance.getTotal());
        
        instance.set("b", 1);
        assertEquals(1, instance.get("b"));
        assertEquals(8, instance.getTotal());
        
        instance.set("c", 4);
        assertEquals(4, instance.get("c"));
        assertEquals(12, instance.getTotal());
        
        instance.set("c", 2);
        assertEquals(2, instance.get("c"));
        assertEquals(10, instance.getTotal());
        
        instance.set("d", 0);
        assertEquals(0, instance.get("d"));
        assertEquals(10, instance.getTotal());
        
        assertEquals(3, instance.getDomain().size());
    }

    /**
     * Test of getTotal method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetTotal()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0, instance.getTotal());
        
        instance.increment("a");
        assertEquals(1, instance.getTotal());
        
        instance.increment("a");
        assertEquals(2, instance.getTotal());

        instance.increment("b");
        assertEquals(3, instance.getTotal());
        
        instance.increment("c", 2);
        assertEquals(5, instance.getTotal());
        
        instance.decrement("b");
        assertEquals(4, instance.getTotal());
        
        instance.decrement("a", 2);
        assertEquals(2, instance.getTotal());
        
        instance.set("c", 0);
        assertEquals(0, instance.getTotal());
    }

    /**
     * Test of clear method, of class DefaultCountDistribution.
     */
    @Test
    public void testClear()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.isEmpty());
        
        instance.increment("a", 2);
        assertFalse(instance.isEmpty());
        
        instance.clear();
        assertTrue(instance.isEmpty());
        assertEquals(0, instance.getTotal());
        assertEquals(0, instance.get("a"));
        
    }

    /**
     * Test of getEstimator method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetEstimator()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertNotNull(instance.getEstimator());
        assertNotSame(instance.getEstimator(), instance.getEstimator());
        DefaultCountDistribution<String> estimated = 
            instance.getEstimator().learn(Arrays.asList("a", "b", "a"));
        assertEquals(3, estimated.getTotal());
        assertEquals(2, estimated.get("a"));
        assertEquals(1, estimated.get("b"));
    }

    /**
     * Test of getProbabilityFunction method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetProbabilityFunction()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0.0, instance.getProbabilityFunction().getTotal(), 0.0);
        assertNotSame(instance.getProbabilityFunction(), instance.getProbabilityFunction());
        
        instance.incrementAll(Arrays.asList("a", "b", "b", "c"));
        assertEquals(0.25, instance.getProbabilityFunction().evaluate("a"), 0.0);
        assertEquals(0.5, instance.getProbabilityFunction().evaluate("b"), 0.0);
        assertEquals(0.25, instance.getProbabilityFunction().evaluate("c"), 0.0);
        assertEquals(0.0, instance.getProbabilityFunction().evaluate("d"), 0.0);
    }

    /**
     * Test of getMeanValue method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetMeanValue()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0.0, instance.getMeanValue(), 0.0);

        instance.increment("a", 2);
        instance.increment("b", 1);
        instance.increment("c", 2);
        instance.increment("d", 1);
        instance.set("e", 0);
        
        assertEquals(1.5, instance.getMeanValue(), 0.0);
    }
    
    @Test
    public void testGetDomain()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.getDomain().isEmpty());
        assertFalse(instance.getDomain().contains("a"));
        assertFalse(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));

        instance.increment("a");
        assertEquals(1, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertFalse(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));

        instance.increment("a");
        assertEquals(1, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertFalse(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));

        instance.increment("b");
        assertEquals(2, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));

        instance.increment("c", 4);
        assertEquals(3, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
        assertTrue(instance.getDomain().contains("c"));

        instance.decrement("a", 2);
        assertEquals(3, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
        assertTrue(instance.getDomain().contains("c"));

        instance.decrement("c", 4);
        assertEquals(3, instance.getDomain().size());
        instance.compact();
        assertFalse(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));

        instance.decrement("b", 1);
        assertEquals(1, instance.getDomain().size());
        assertFalse(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));
        instance.compact();
        assertFalse(instance.getDomain().contains("a"));
        assertFalse(instance.getDomain().contains("b"));
        assertFalse(instance.getDomain().contains("c"));
    }
    
    /**
     * Test of getDomainSize method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetDomainSize()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0, instance.getDomainSize());

        instance.increment("a");
        assertEquals(1, instance.getDomainSize());

        instance.increment("a");
        assertEquals(1, instance.getDomainSize());

        instance.increment("b");
        assertEquals(2, instance.getDomainSize());

        instance.increment("c", 4);
        assertEquals(3, instance.getDomainSize());

        instance.decrement("a", 2);
        assertEquals(3, instance.getDomainSize());

        instance.decrement("c", 4);
        assertEquals(3, instance.getDomainSize());

        instance.decrement("b", 1);
        assertEquals(3, instance.getDomainSize());
        instance.compact();
        assertEquals(0, instance.getDomainSize());
    }
    
    /**
     * Test of isEmpty method, of class DefaultCountDistribution.
     */
    @Test
    public void testIsEmpty()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.isEmpty());

        instance.increment("a");
        assertFalse(instance.isEmpty());
        
        instance.increment("a");
        assertFalse(instance.isEmpty());
                
        instance.increment("b");
        assertFalse(instance.isEmpty());
        
        instance.decrement("a", 2);
        assertFalse(instance.isEmpty());
        
        instance.decrement("b", 1);
        assertFalse(instance.isEmpty());
        
        instance.compact();
        assertTrue(instance.isEmpty());
    }
    
    @Test
    public void testGetFraction()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(0.0, instance.getFraction("a"));
        assertEquals(0.0, instance.getFraction("b"));
        assertEquals(0.0, instance.getFraction("c"));
        assertEquals(0.0, instance.getFraction("d"));

        instance.increment("a");
        assertEquals(1 / 1.0, instance.getFraction("a"));

        instance.increment("a");
        assertEquals(2 / 2.0, instance.getFraction("a"));

        instance.increment("b");
        assertEquals(2 / 3.0, instance.getFraction("a"));
        assertEquals(1 / 3.0, instance.getFraction("b"));

        instance.increment("c", 4);
        assertEquals(2 / 7.0, instance.getFraction("a"));
        assertEquals(1 / 7.0, instance.getFraction("b"));
        assertEquals(4 / 7.0, instance.getFraction("c"));

        instance.increment("a", 2);
        assertEquals(4 / 9.0, instance.getFraction("a"));
        assertEquals(1 / 9.0, instance.getFraction("b"));
        assertEquals(4 / 9.0, instance.getFraction("c"));

        instance.decrement("a");
        assertEquals(3 / 8.0, instance.getFraction("a"));
        assertEquals(1 / 8.0, instance.getFraction("b"));
        assertEquals(4 / 8.0, instance.getFraction("c"));

        instance.decrement("c", 3);
        assertEquals(3 / 5.0, instance.getFraction("a"));
        assertEquals(1 / 5.0, instance.getFraction("b"));
        assertEquals(1 / 5.0, instance.getFraction("c"));

        instance.decrement("b", 1);
        assertEquals(3 / 4.0, instance.getFraction("a"));
        assertEquals(0 / 4.0, instance.getFraction("b"));
        assertEquals(1 / 4.0, instance.getFraction("c"));

        instance.increment("d");
        assertEquals(3 / 5.0, instance.getFraction("a"));
        assertEquals(0 / 5.0, instance.getFraction("b"));
        assertEquals(1 / 5.0, instance.getFraction("c"));
        assertEquals(1 / 5.0, instance.getFraction("d"));
    }
    
    
    /**
     * Test of getLogFraction method, of class DefaultCountDistribution.
     */
    @Test
    public void testGetLogFraction()
    {
        double epsilon = 1e-10;
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertEquals(Double.NEGATIVE_INFINITY, instance.getLogFraction("a"));
        assertEquals(Double.NEGATIVE_INFINITY, instance.getLogFraction("b"));
        assertEquals(Double.NEGATIVE_INFINITY, instance.getLogFraction("c"));
        assertEquals(Double.NEGATIVE_INFINITY, instance.getLogFraction("d"));

        instance.increment("a", 4);
        instance.increment("b");
        instance.increment("c", 4);
        assertEquals(Math.log(4 / 9.0), instance.getLogFraction("a"), epsilon);
        assertEquals(Math.log(1 / 9.0), instance.getLogFraction("b"), epsilon);
        assertEquals(Math.log(4 / 9.0), instance.getLogFraction("c"), epsilon);
        assertEquals(Double.NEGATIVE_INFINITY, instance.getLogFraction("d"));
    }
    
    @Test
    public void testGetEntropy()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();

        assertEquals(0.0, instance.getEntropy());

        instance.increment("a");
        assertEquals(0.0, instance.getEntropy());

        instance.increment("a");
        assertEquals(0.0, instance.getEntropy());

        instance.increment("b");
        assertEquals(0.9183, instance.getEntropy(), 0.0001);

        instance.increment("c");
        assertEquals(1.5000, instance.getEntropy(), 0.0001);

        instance.increment("c", 4);
        assertEquals(1.2988, instance.getEntropy(), 0.0001);

        instance.increment("d", 1);
        assertEquals(1.6577, instance.getEntropy(), 0.0001);
    }
    
    @Test
    public void testGetMinValue()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        
        assertEquals(0, instance.getMinValue());

        instance.increment("a");
        assertEquals(1, instance.getMinValue());
        instance.increment("b");
        assertEquals(1, instance.getMinValue());
        instance.increment("b");
        assertEquals(1, instance.getMinValue());
        instance.increment("a", 3);
        assertEquals(2, instance.getMinValue());
    }
    
    @Test
    public void testGetMaxValue()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        
        assertEquals(0, instance.getMaxValue());

        instance.increment("a");
        assertEquals(1, instance.getMaxValue());
        instance.increment("b");
        assertEquals(1, instance.getMaxValue());
        instance.increment("b");
        assertEquals(2, instance.getMaxValue());
        instance.increment("c", 7);
        assertEquals(7, instance.getMaxValue());
    }
    
    @Test
    public void testGetMinValueKey()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertNull(instance.getMinValueKey());

        instance.increment("a");
        assertEquals("a", instance.getMinValueKey());
        instance.increment("b");
        assertTrue("a".equals(instance.getMinValueKey())); // a should be the first value encountered.
        instance.increment("b");
        assertEquals("a", instance.getMinValueKey());
        instance.increment("a", 3);
        assertEquals("b", instance.getMinValueKey());
    }
    
    @Test
    public void testGetMaxValueKey()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertNull(instance.getMaxValueKey());

        instance.increment("a");
        assertEquals("a", instance.getMaxValueKey());
        instance.increment("b");
        assertTrue("a".equals(instance.getMaxValueKey())); // a should be the first value encountered.
        instance.increment("b");
        assertEquals("b", instance.getMaxValueKey());
        instance.increment("c", 7);
        assertEquals("c", instance.getMaxValueKey());
    }
    
    @Test
    public void testGetMinValueKeys()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.getMinValueKeys().isEmpty());

        instance.increment("a");
        assertEquals(1, instance.getMinValueKeys().size());
        assertTrue(instance.getMinValueKeys().contains("a"));
        instance.increment("b");
        instance.increment("c", 7);
        assertEquals(2, instance.getMinValueKeys().size());
        assertTrue(instance.getMinValueKeys().contains("a"));
        assertTrue(instance.getMinValueKeys().contains("b"));
        instance.increment("a");
        assertEquals(1, instance.getMinValueKeys().size());
        assertTrue(instance.getMinValueKeys().contains("b"));
    }
    
    @Test
    public void testGetMaxValueKeys()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertTrue(instance.getMaxValueKeys().isEmpty());

        instance.increment("a");
        assertEquals(1, instance.getMaxValueKeys().size());
        assertTrue(instance.getMaxValueKeys().contains("a"));
        instance.increment("b");
        assertEquals(2, instance.getMaxValueKeys().size());
        assertTrue(instance.getMaxValueKeys().contains("a"));
        assertTrue(instance.getMaxValueKeys().contains("b"));
        instance.increment("b");
        assertEquals(1, instance.getMaxValueKeys().size());
        assertTrue(instance.getMaxValueKeys().contains("b"));
        instance.increment("c", 7);
        assertEquals(1, instance.getMaxValueKeys().size());
        assertTrue(instance.getMaxValueKeys().contains("c"));
    }
    
    
    /**
     * Test of toInfiniteVector method, of class DefaultCountDistribution.
     */
    @Test
    public void testToInfiniteVector()
    {
        DefaultCountDistribution<String> instance = this.createEmpty();
        InfiniteVector<String> result = instance.toInfiniteVector();
        assertTrue(result.isEmpty());
        
        instance = this.createExample();
        result = instance.toInfiniteVector();
        assertEquals(4.0, result.sum());
        assertEquals(2.0, result.get("a"));
        assertEquals(1.0, result.get("b"));
        assertEquals(1.0, result.get("c"));
    }

    /**
     * Test of fromInfiniteVector method, of class DefaultCountDistribution.
     */
    @Test
    public void testFromInfiniteVector()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        
        DefaultInfiniteVector<String> vector = new DefaultInfiniteVector<>();
        vector.set("a", 3);
        vector.set("c", 2);
        
        instance.fromInfiniteVector(vector);
        assertEquals(5, instance.getTotal());
        assertEquals(3, instance.get("a"));
        assertEquals(0, instance.get("b"));
        assertEquals(2, instance.get("c"));
    }
    
    /**
     * Test of sample method, of class DefaultCountDistribution.
     */
    @Test
    public void testSample()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();
        assertNull(instance.sample(random));
        
        instance.increment("a");
        assertEquals("a", instance.sample(random));
        assertEquals(Arrays.asList("a", "a"), instance.sample(random, 2));
        
        instance.increment("a");
        instance.increment("b");
        instance.increment("c", 2);
        
        DefaultCountDistribution<String> samples = new DefaultCountDistribution<>();
        samples.incrementAll(instance.sample(random, 10000));
        assertEquals(0.4, samples.getFraction("a"), 0.01);
        assertEquals(0.2, samples.getFraction("b"), 0.01);
        assertEquals(0.4, samples.getFraction("c"), 0.01);
    }

    /**
     * Test of sampleInto method, of class DefaultCountDistribution.
     */
    @Test
    public void testSampleInto()
    {
        DefaultCountDistribution<String> instance = new DefaultCountDistribution<>();

        List<String> output = new LinkedList<>();
        instance.sampleInto(random, 2, output);
        assertEquals(2, output.size());
        assertNull(output.get(0));
        assertNull(output.get(1));

        instance.increment("a");
        instance.decrement("a");
        output.clear();
        instance.sampleInto(random, 1, output);
        assertEquals(1, output.size());
        assertNull(output.get(0));
        
        instance.increment("a");
        output.clear();
        instance.sampleInto(random, 2, output);
        assertEquals(Arrays.asList("a", "a"), output);
        
        instance.increment("a");
        instance.increment("b");
        instance.increment("c", 2);
        
        output.clear();
        instance.sampleInto(random, 10000, output);
        DefaultCountDistribution<String> samples = new DefaultCountDistribution<>(output);
        assertEquals(0.4, samples.getFraction("a"), 0.01);
        assertEquals(0.2, samples.getFraction("b"), 0.01);
        assertEquals(0.4, samples.getFraction("c"), 0.01);
    }
    
    @Test
    public void testEvaluatePMF()
    {
        CountDistribution.PMF<String> instance = new DefaultCountDistribution<String>().getProbabilityFunction();
        assertEquals(0.0, instance.evaluate("a"));
        assertEquals(0.0, instance.evaluate("b"));
        assertEquals(0.0, instance.evaluate("c"));
        assertEquals(0.0, instance.evaluate("d"));

        instance.increment("a");
        assertEquals(1 / 1.0, instance.evaluate("a"));

        instance.increment("a");
        assertEquals(2 / 2.0, instance.evaluate("a"));

        instance.increment("b");
        assertEquals(2 / 3.0, instance.evaluate("a"));
        assertEquals(1 / 3.0, instance.evaluate("b"));

        instance.increment("c", 4);
        assertEquals(2 / 7.0, instance.evaluate("a"));
        assertEquals(1 / 7.0, instance.evaluate("b"));
        assertEquals(4 / 7.0, instance.evaluate("c"));

        instance.increment("a", 2);
        assertEquals(4 / 9.0, instance.evaluate("a"));
        assertEquals(1 / 9.0, instance.evaluate("b"));
        assertEquals(4 / 9.0, instance.evaluate("c"));

        instance.decrement("a");
        assertEquals(3 / 8.0, instance.evaluate("a"));
        assertEquals(1 / 8.0, instance.evaluate("b"));
        assertEquals(4 / 8.0, instance.evaluate("c"));

        instance.decrement("c", 3);
        assertEquals(3 / 5.0, instance.evaluate("a"));
        assertEquals(1 / 5.0, instance.evaluate("b"));
        assertEquals(1 / 5.0, instance.evaluate("c"));

        instance.decrement("b", 1);
        assertEquals(3 / 4.0, instance.evaluate("a"));
        assertEquals(0 / 4.0, instance.evaluate("b"));
        assertEquals(1 / 4.0, instance.evaluate("c"));

        instance.increment("d");
        assertEquals(3 / 5.0, instance.evaluate("a"));
        assertEquals(0 / 5.0, instance.evaluate("b"));
        assertEquals(1 / 5.0, instance.evaluate("c"));
        assertEquals(1 / 5.0, instance.evaluate("d"));
    }
    
    @Test
    public void testLogEvaluatePMF()
    {
        CountDistribution.PMF<String> instance = new DefaultCountDistribution<String>().getProbabilityFunction();
        
        for( String s : instance.getDomain() )
        {
            double plog = instance.logEvaluate(s);
            double p = instance.evaluate(s);
            double phat = Math.exp(plog);
            assertEquals( p, phat, 1e-5 );
        }

    }
    
    @Test
    public void testLearnViaEstimator()
    {
        DefaultCountDistribution.Estimator<String> learner =
            new DefaultCountDistribution.Estimator<String>();
        ArrayList<String> data = new ArrayList<String>();

        DefaultCountDistribution<String> instance = learner.learn(data);
        assertEquals(0, instance.getTotal());
        assertEquals(0, instance.get("a"));
        assertEquals(0, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(0, instance.getDomain().size());
        assertFalse(instance.getDomain().contains("a"));

        data.add("a");
        instance = learner.learn(data);
        assertEquals(1, instance.getTotal());
        assertEquals(1, instance.get("a"));
        assertEquals(0, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(1, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));

        data.add("a");
        instance = learner.learn(data);
        assertEquals(2, instance.getTotal());
        assertEquals(2, instance.get("a"));
        assertEquals(0, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(1, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));

        data.add("b");
        instance = learner.learn(data);
        assertEquals(3, instance.getTotal());
        assertEquals(2, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(2, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));

        data.add("a");
        instance = learner.learn(data);
        assertEquals(4, instance.getTotal());
        assertEquals(3, instance.get("a"));
        assertEquals(1, instance.get("b"));
        assertEquals(0, instance.get("c"));
        assertEquals(2, instance.getDomain().size());
        assertTrue(instance.getDomain().contains("a"));
        assertTrue(instance.getDomain().contains("b"));
    }
}
