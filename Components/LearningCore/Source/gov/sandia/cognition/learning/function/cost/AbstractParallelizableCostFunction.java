/*
 * File:                AbstractParallelizableCostFunction.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 * 
 * Copyright Sep 23, 2008, Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government. 
 * Export of this program may require a license from the United States
 * Government. See CopyrightHistory.txt for complete details.
 * 
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.math.matrix.Vector;
import java.util.Arrays;
import java.util.Collection;

/**
 * Partial implementation of the ParallelizableCostFunction
 * @author Kevin R. Dixon
 * @since 2.1
 * @param <InputType> The type of input for the evaluated function.
 * @param <OutputType> The type of output for the evaluated function.
 * @param <EvaluatedType> The type of evaluated function.
 * @param <DifferentiableEvaluatedType> The type of evaluated function that
 *      can be differentiated.
 */
public abstract class AbstractParallelizableCostFunction<InputType, OutputType, EvaluatedType extends Evaluator<? super InputType, ? extends OutputType>, DifferentiableEvaluatedType extends EvaluatedType>
    extends AbstractSupervisedCostFunction<InputType, OutputType, EvaluatedType>
    implements ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType>
{

    /** 
     * Creates a new instance of AbstractParallelizableCostFunction 
     * @param costParameters 
     * Dataset to use
     */
    public AbstractParallelizableCostFunction(
        Collection<? extends InputOutputPair<? extends InputType, OutputType>> costParameters )
    {
        super( costParameters );
    }

    @Override
    public double evaluateAsDouble(
        EvaluatedType evaluator )
    {
        Object result = this.evaluatePartial( evaluator );
        return this.evaluateAmalgamate( Arrays.asList( result ) );
    }

    @Override
    public double computeCost(
        final DifferentiableEvaluatedType function)
    {
        return this.evaluateAsDouble(function);
    }
    
    @Override
    public Vector computeParameterGradient(
        DifferentiableEvaluatedType function )
    {
        Object result = this.computeParameterGradientPartial( function );
        return this.computeParameterGradientAmalgamate( Arrays.asList( result ) );
    }

    @SuppressWarnings("unchecked")
    @Override
    public AbstractParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> clone()
    {
        return (AbstractParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType>) super.clone();
    }
  
}
