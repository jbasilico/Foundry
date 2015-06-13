/*
 * File:                SupervisedCostFunction.java
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
 * 
 */

package gov.sandia.cognition.learning.function.cost;

import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.learning.data.TargetEstimatePair;
import gov.sandia.cognition.learning.performance.SupervisedPerformanceEvaluator;
import gov.sandia.cognition.util.Summarizer;
import java.util.Collection;

/**
 * A type of CostFunction normally used in supervised-learning applications.
 * This CostFunction evaluates an Evaluator against its ability to reproduce
 * the targets from a collection of InputOutputPairs
 * 
 * @param <InputType> Inputs of the Evaluator, such as a Vector
 * @param <TargetType> Targets of the Evaluator, such a Double
 * @param <EvaluatedType> Type of the evaluator to compute the cost for.
 * @author Kevin R. Dixon
 * @since 2.0
 */
public interface SupervisedCostFunction<InputType, TargetType, EvaluatedType extends Evaluator<? super InputType, ? extends TargetType>>
    extends CostFunction<EvaluatedType, Collection<? extends InputOutputPair<? extends InputType, TargetType>>>,
    Summarizer<TargetEstimatePair<? extends TargetType, ? extends TargetType>, Double>,
    SupervisedPerformanceEvaluator<InputType, TargetType, TargetType, Double>
{
    
    /**
     * Evaluate the performance of this cost function as a double from a
     * given collection of target-estimate pairs.
     * 
     * @param   data
     *      The collection of target-estimate pairs to evaluate.
     * @return 
     *      The cost computed from the target-estimate pairs
     */
    public abstract double evaluatePerformanceAsDouble(
        final Collection<? extends TargetEstimatePair<? extends TargetType, ? extends TargetType>> data);
}
