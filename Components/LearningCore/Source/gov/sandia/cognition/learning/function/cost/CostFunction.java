/*
 * File:                CostFunction.java
 * Authors:             Justin Basilico and Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 *
 * Copyright February 20, 2006, Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this work by
 * or on behalf of the U.S. Government. Export of this program may require a
 * license from the United States Government. See CopyrightHistory.txt for
 * complete details.
 *
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.annotation.CodeReview;
import gov.sandia.cognition.math.ScalarFunction;

/**
 * The CostFunction interface defines the interface to evaluate some object to
 * determine its cost. The interface defines the ability to get the set of
 * parameters used for the cost function and to set them. This is used to
 * facilitate learning algorithms passing the data to evaluate the cost of
 * a hypothesis on.
 * 
 * @param <EvaluatedType>
 * Class type to evaluate, for example a "Vector" or "Evaluator" 
 * @param <CostParametersType>
 * Class type that parameterizes the CostFunction, for example, a Collection of
 * InputOutputPairs.  Usually the dataset we're interested in.
 * @author Justin Basilico
 * @author Kevin R. Dixon
 * @since 1.0
 */
@CodeReview(
    reviewer="Justin Basilico",
    date="2006-10-04",
    changesNeeded=false,
    comments="Interface looks fine."
)
public interface CostFunction<EvaluatedType, CostParametersType>
    extends ScalarFunction<EvaluatedType>, CostParameterized<CostParametersType>
{
    /**
     * Computes the cost of the given target.
     * 
     * @param target The object to evaluate.
     * @return The cost of the given object.
     */
    @Override
    public Double evaluate(
        EvaluatedType target);
    
    /**
     * Computes the cost of the given target.
     * 
     * @param target The object to evaluate.
     * @return The cost of the given object.
     */
    @Override
    public double evaluateAsDouble(
        EvaluatedType target);
    
}
