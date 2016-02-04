/*
 * File:            ClipFunction.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2016 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.function.scalar;

import gov.sandia.cognition.math.AbstractDifferentiableUnivariateScalarFunction;

/**
 * A function that clips its input to be between a minimum and maximum values.
 * That is, f(x) = max(l, min(x, u)) where l is the lower bound and u is
 * the upper bound. Between l and u this is an identity function.
 * 
 * @author  Justin Basilico
 * @since   3.4.3
 */
public class ClipFunction
    extends AbstractDifferentiableUnivariateScalarFunction
{
    
    /** The default minimum value is {@value}. */
    public static final double DEFAULT_MIN_VALUE = Double.NEGATIVE_INFINITY;
    
    /** The default maximum value is {@value}. */
    public static final double DEFAULT_MAX_VALUE = Double.POSITIVE_INFINITY;
    
    /** The minimum value. Should be less than the max value. */
    protected double minValue;
    
    /** The maximum value. Should be more than the min value. */
    protected double maxValue;

    /**
     * Creates a new {@link ClipFunction} with default parameters.
     */
    public ClipFunction()
    {
        this(DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE);
    }

    /**
     * Creates a new {@link ClipFunction} with the given parameters.
     * 
     * @param   minValue
     *      The minimum value. Should be less than the max value.
     * @param   maxValue 
     *      The maximum value. Should be more than the min value. 
     */
    public ClipFunction(
        final double minValue,
        final double maxValue)
    {
        super();
        
        this.setMinValue(minValue);
        this.setMaxValue(maxValue);
    }

    @Override
    public ClipFunction clone()
    {
        return (ClipFunction) super.clone();
    }
    
    @Override
    public double evaluate(
        final double input)
    {
        if (input < this.minValue)
        {
            return this.minValue;
        }
        else if (input > this.maxValue)
        {
            return this.maxValue;
        }
        else
        {
            return input;
        }
    }

    @Override
    public double differentiate(
        final double input)
    {
        if (input < this.minValue)
        {
            return 0.0;
        }
        else if (input > this.maxValue)
        {
            return 0.0;
        }
        else
        {
            return 1.0;
        }
    }

    /**
     * Gets the minimum value.
     * 
     * @return 
     *      The minimum value.
     */
    public double getMinValue()
    {
        return this.minValue;
    }

    /**
     * Sets the minimum value.
     * 
     * @param   minValue 
     *      The minimum value.
     */
    public void setMinValue(
        final double minValue)
    {
        this.minValue = minValue;
    }

    /**
     * Gets the maximum value.
     * 
     * @return 
     *      The maximum value.
     */
    public double getMaxValue()
    {
        return this.maxValue;
    }

    /**
     * Sets the maximum value.
     * 
     * @param   maxValue 
     *      The maximum value.
     */
    public void setMaxValue(
        final double maxValue)
    {
        this.maxValue = maxValue;
    }
    
}
