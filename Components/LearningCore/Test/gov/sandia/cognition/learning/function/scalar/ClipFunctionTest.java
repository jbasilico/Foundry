/*
 * File:            ClipFunctionTest.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2016 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.function.scalar;

import java.util.Random;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for class {@link ClipFunction}.
 * 
 * @author  Justin Basilico
 * @since   3.4.3
 */
public class ClipFunctionTest
    extends Object
{
    protected Random random = new Random(1239799);
    
    /**
     * Creates a new test.
     */
    public ClipFunctionTest()
    {
    }

    /**
     * Test of constructor of class ClipFunction.
     */
    @Test
    public void testConstructor()
    {
        double minValue = ClipFunction.DEFAULT_MIN_VALUE;
        double maxValue = ClipFunction.DEFAULT_MAX_VALUE;
        ClipFunction instance = new ClipFunction();
        assertEquals(minValue, instance.getMinValue(), 0.0);
        assertEquals(maxValue, instance.getMaxValue(), 0.0);
        
        minValue = this.random.nextGaussian();
        maxValue = minValue + this.random.nextDouble();
        instance = new ClipFunction(minValue, maxValue);
        assertEquals(minValue, instance.getMinValue(), 0.0);
        assertEquals(maxValue, instance.getMaxValue(), 0.0);
    }
    
    /**
     * Test of clone method, of class ClipFunction.
     */
    @Test
    public void testClone()
    {
        ClipFunction instance = new ClipFunction(3.0, 4.1);
        ClipFunction clone = instance.clone();
        assertNotSame(clone, instance);
        assertEquals(3.0, clone.getMinValue(), 0.0);
        assertEquals(4.1, clone.getMaxValue(), 0.0);
        assertNotSame(clone, instance.clone());
    }

    /**
     * Test of evaluate method, of class ClipFunction.
     */
    @Test
    public void testEvaluate()
    {
        ClipFunction instance = new ClipFunction();
        for (int i = 0; i < 10; i++)
        {
            double input = 100.0 * this.random.nextGaussian();
            assertEquals(input, instance.evaluate(input), 0.0);
        }
        
        instance.setMinValue(-4.0);
        instance.setMaxValue(3.0);
        
        assertEquals(-4.0, instance.evaluate(-10.0), 0.0);
        assertEquals(-4.0, instance.evaluate(-4.0), 0.0);
        assertEquals(2.2, instance.evaluate(2.2), 0.0);
        assertEquals(3.0, instance.evaluate(3.0), 0.0);
        assertEquals(3.0, instance.evaluate(4.0), 0.0);
    }

    /**
     * Test of differentiate method, of class ClipFunction.
     */
    @Test
    public void testDifferentiate()
    {
        ClipFunction instance = new ClipFunction();
        for (int i = 0; i < 10; i++)
        {
            double input = 100.0 * this.random.nextGaussian();
            assertEquals(1.0, instance.differentiate(input), 0.0);
        }
        
        instance.setMinValue(-4.0);
        instance.setMaxValue(3.0);
        
        assertEquals(0.0, instance.differentiate(-10.0), 0.0);
        assertEquals(0.0, instance.differentiate(-4.1), 0.0);
        assertEquals(1.0, instance.differentiate(-4.0), 0.0);
        assertEquals(1.0, instance.differentiate(-3.9), 0.0);
        assertEquals(1.0, instance.differentiate(2.2), 0.0);
        assertEquals(1.0, instance.differentiate(2.99), 0.0);
        assertEquals(1.0, instance.differentiate(3.0), 0.0);
        assertEquals(0.0, instance.differentiate(3.1), 0.0);
        assertEquals(0.0, instance.differentiate(4.0), 0.0);
    }

    /**
     * Test of getMinValue method, of class ClipFunction.
     */
    @Test
    public void testGetMinValue()
    {
        this.testSetMinValue();
    }

    /**
     * Test of setMinValue method, of class ClipFunction.
     */
    @Test
    public void testSetMinValue()
    {
        double minValue = ClipFunction.DEFAULT_MIN_VALUE;
        ClipFunction instance = new ClipFunction();
        assertEquals(minValue, instance.getMinValue(), 0.0);
        
        minValue = this.random.nextGaussian();
        instance.setMinValue(minValue);
        assertEquals(minValue, instance.getMinValue(), 0.0);
    }

    /**
     * Test of getMaxValue method, of class ClipFunction.
     */
    @Test
    public void testGetMaxValue()
    {
        this.testSetMaxValue();
    }

    /**
     * Test of setMaxValue method, of class ClipFunction.
     */
    @Test
    public void testSetMaxValue()
    {
        double maxValue = ClipFunction.DEFAULT_MAX_VALUE;
        ClipFunction instance = new ClipFunction();
        assertEquals(maxValue, instance.getMaxValue(), 0.0);
        
        maxValue = this.random.nextGaussian();
        instance.setMaxValue(maxValue);
        assertEquals(maxValue, instance.getMaxValue(), 0.0);
    }
    
}
