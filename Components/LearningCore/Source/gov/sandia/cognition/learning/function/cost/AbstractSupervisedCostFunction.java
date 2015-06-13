/*
 * File:                AbstractSupervisedCostFunction.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 * 
 * Copyright Dec 20, 2007, Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this work by
 * or on behalf of the U.S. Government. Export of this program may require a
 * license from the United States Government. See CopyrightHistory.txt for
 * complete details.
 * 
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.learning.data.DatasetUtil;
import gov.sandia.cognition.learning.data.DefaultWeightedTargetEstimatePair;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.learning.data.TargetEstimatePair;
import gov.sandia.cognition.learning.data.WeightedTargetEstimatePair;
import gov.sandia.cognition.learning.performance.AbstractSupervisedPerformanceEvaluator;
import gov.sandia.cognition.util.ObjectUtil;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Abstract class for {@link SupervisedCostFunction}. Defines the basic common
 * implementation, like saving the cost parameters.
 * 
 * @param <InputType> Input type of the dataset and Evaluator
 * @param <TargetType> Output type (labels) of the dataset and Evaluator
 * @param <EvaluatedType> The type of evaluator to compute the cost for.
 * @author Kevin R. Dixon
 * @since 2.0
 */
public abstract class AbstractSupervisedCostFunction<InputType, TargetType, EvaluatedType extends Evaluator<? super InputType, ? extends TargetType>>
    extends AbstractSupervisedPerformanceEvaluator<InputType, TargetType, TargetType, Double>
    implements SupervisedCostFunction<InputType, TargetType, EvaluatedType>
{

    /**
     * Labeled dataset to use to evaluate the cost against.
     */
    protected Collection<? extends InputOutputPair<? extends InputType, TargetType>> costParameters;

    /** 
     * Creates a new {@link AbstractSupervisedCostFunction}. The cost parameters
     * are initialized to null.
     */
    public AbstractSupervisedCostFunction()
    {
        this(null);
    }

    /**
     * Creates a new {@link AbstractSupervisedCostFunction} with the given
     * cost parameters.
     * 
     * @param costParameters
     *      Labeled dataset to use to evaluate the cost against.
     */
    public AbstractSupervisedCostFunction(
        final Collection<? extends InputOutputPair<? extends InputType, TargetType>> costParameters)
    {
        super();
        
        this.setCostParameters(costParameters);
    }

    @Override
    @SuppressWarnings("unchecked")
    public AbstractSupervisedCostFunction<InputType, TargetType, EvaluatedType> clone()
    {
        final AbstractSupervisedCostFunction<InputType, TargetType, EvaluatedType> clone =
            (AbstractSupervisedCostFunction<InputType, TargetType, EvaluatedType>) super.clone();
        clone.setCostParameters(
            ObjectUtil.cloneSmartElementsAsArrayList(this.getCostParameters()));
        return clone;
    }

    @Override
    public Double evaluatePerformance(
        final Collection<? extends TargetEstimatePair<? extends TargetType, ? extends TargetType>> data)
    {
        return this.evaluatePerformanceAsDouble(data);
    }
    
    @Override
    public abstract double evaluatePerformanceAsDouble(
        final Collection<? extends TargetEstimatePair<? extends TargetType, ? extends TargetType>> data);
    
    @Override
    public Double evaluate(
        final EvaluatedType evaluator)
    {
        return this.evaluateAsDouble(evaluator);
    }
    
    @Override
    public double evaluateAsDouble(
        final EvaluatedType evaluator)
    {
        final ArrayList<WeightedTargetEstimatePair<TargetType, TargetType>> targetEstimatePairs = 
            new ArrayList<WeightedTargetEstimatePair<TargetType, TargetType>>(
                this.getCostParameters().size());

        for (final InputOutputPair<? extends InputType, ? extends TargetType> io
            : this.getCostParameters())
        {
            final TargetType target = io.getOutput();
            final TargetType estimate = evaluator.evaluate(io.getInput());
            targetEstimatePairs.add(DefaultWeightedTargetEstimatePair.create(
                target, estimate, DatasetUtil.getWeight(io)));
        }

        return this.evaluatePerformanceAsDouble(targetEstimatePairs);
    }

    @Override
    public Double summarize(
        final Collection<? extends TargetEstimatePair<? extends TargetType, ? extends TargetType>> data)
    {
        return this.evaluatePerformance(data);
    }
    
    @Override
    public Collection<? extends InputOutputPair<? extends InputType, TargetType>> getCostParameters()
    {
        return this.costParameters;
    }

    @Override
    public void setCostParameters(
        Collection<? extends InputOutputPair<? extends InputType, TargetType>> costParameters)
    {
        this.costParameters = costParameters;
    }

}
