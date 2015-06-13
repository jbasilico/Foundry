/*
 * File:                ParallelizableCostFunction.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 * 
 * Copyright Sep 22, 2008, Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government. 
 * Export of this program may require a license from the United States
 * Government. See CopyrightHistory.txt for complete details.
 * 
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.math.matrix.Vector;
import java.util.Collection;

/**
 * Defines a differentiable cost function that can (largely) be computed in
 * parallel.
 * 
 * @author Kevin R. Dixon
 * @since 2.1
 * @param <InputType> The type of input for the evaluated function.
 * @param <OutputType> The type of output for the evaluated function.
 * @param <EvaluatedType> The type of evaluated function.
 * @param <DifferentiableEvaluatedType> The type of evaluated function that
 *      can be differentiated.
 */
public interface ParallelizableCostFunction<InputType, OutputType, EvaluatedType extends Evaluator<? super InputType, ? extends OutputType>, DifferentiableEvaluatedType extends EvaluatedType>
    extends SupervisedCostFunction<InputType, OutputType, EvaluatedType>, 
        DifferentiableCostFunction<InputType, OutputType, DifferentiableEvaluatedType>
{
// TODO: This currently forces all parallelizable cost functions to be differentiable. Should this be loosened or perhaps another interface introduced for differentiation?
    
// TODO: The results of the parallelization seem to just be Object here. Would be nice if there was a way to preserve type safety.
    /**
     * Computes the partial (linear) component of the cost function.
     * This portion will be performed in parallel.
     * @param evaluator
     * Evaluator to compute the cost of
     * @return
     * Object that contains the linear component of the cost function
     */
    public Object evaluatePartial(
        EvaluatedType evaluator);

    /**
     * Amalgamates the linear components of the cost function into a single
     * Double. This portion will be performed in sequence.
     * @param partialResults
     * Collection of partial (linear) results
     * @return
     * Cost function of the partial results
     */
    public Double evaluateAmalgamate(
        Collection<Object> partialResults);
    
    /**
     * Computes the partial (linear) component of the cost function gradient.
     * This portion will be performed in parallel.
     * 
     * @param function
     * GradientDescendable to compute the gradient of
     * @return
     * Object that contains the linear component of the gradient
     */
    public Object computeParameterGradientPartial(
        DifferentiableEvaluatedType function);

    /**
     * Amalgamates the linear components of the cost gradient function into a
     * single Vector. This portion will be performed in sequence.
     * @param partialResults
     * Collection of partial (linear) gradient components
     * @return
     * Vector describing the gradient
     */
    public Vector computeParameterGradientAmalgamate(
        Collection<Object> partialResults);

}
