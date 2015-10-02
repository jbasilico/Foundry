/*
 * File:            CostParameterized.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.function.cost;

/**
 * Interface for a cost function that has parameters.
 * 
 * @param   <CostParametersType>
 *      The type of cost parameters.
 * @author  Justin Basilico
 * @since   3.4.2
 */
public interface CostParameterized<CostParametersType>
{
    
    /**
     * Sets the parameters of the cost function used to evaluate the cost of
     * a target.
     *
     * @param  costParameters The parameters of the cost function.
     */
    public void setCostParameters(
        CostParametersType costParameters);
    
    /**
     * Gets the parameters of the cost function.
     *
     * @return The current parameters of the cost function.
     */
    public CostParametersType getCostParameters();
    
}
