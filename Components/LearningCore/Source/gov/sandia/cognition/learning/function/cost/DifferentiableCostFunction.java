/*
 * File:                DifferentiableCostFunction.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Framework Lite
 *
 * Copyright April 26, 2006, Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this work by
 * or on behalf of the U.S. Government. Export of this program may require a
 * license from the United States Government. See CopyrightHistory.txt for
 * complete details.
 *
 *
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.math.matrix.Vector;

/**
 * Defines a cost function that can be differentiated. This requires that it 
 * operate as a cost function and it has a separate method for
 * doing the differentiation of a given type of evaluator with respect to the 
 * cost function.
 *
 * @param <InputType> The type of input data for the differentiable function.
 * @param <OutputType> The type of output for the differentiable function.
 * @param <EvaluatedType> Type of evaluator (function) to evaluate and 
 *      differentiate.
 * @author Kevin R. Dixon
 * @since  1.0
 */
public interface DifferentiableCostFunction<InputType, OutputType, EvaluatedType extends Evaluator<? super InputType, ? extends OutputType>>
{
 
    /**
     * Computes the cost for a given differentiable type. Normally when
     * both {@link CostFunction} and {@link DifferentiableCostFunction} are
     * implemented by the same class, this is just an alias for {@code evaluate}
     * in {@link CostFunction}. It exists because to differentiate, the
     * {@link DifferentiableCostFunction} interface may need a more restrictive
     * type than what is required than just computing a {@link CostFunction}.
     * Thus, rather than having two generics for the evaluated type, this
     * sticks with one for the general type and one for the differentiated
     * type.
     * 
     * @param   function
     *      The function to evaluate the cost for.
     * @return 
     *      The cost for the given function.
     */
    public double computeCost(
        final EvaluatedType function);
    
    /**
     * Differentiates function with respect to its parameters.
     *
     * @param function The object to differentiate.
     * @return Derivatives of the object with respect to the cost function.
     */
    public Vector computeParameterGradient(
        final EvaluatedType function);
    
}
