/*
 * File:                ParameterDifferentiableCostMinimizer.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 * 
 * Copyright Jul 4, 2008, Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government. 
 * Export of this program may require a license from the United States
 * Government. See CopyrightHistory.txt for complete details.
 * 
 */

package gov.sandia.cognition.learning.algorithm.regression;

import gov.sandia.cognition.learning.algorithm.gradient.GradientDescendable;
import gov.sandia.cognition.learning.algorithm.minimization.FunctionMinimizer;
import gov.sandia.cognition.learning.algorithm.minimization.FunctionMinimizerBFGS;
import gov.sandia.cognition.learning.algorithm.minimization.line.LineMinimizerBacktracking;
import gov.sandia.cognition.learning.function.cost.DifferentiableCostFunction;
import gov.sandia.cognition.math.AbstractScalarFunction;
import gov.sandia.cognition.math.DifferentiableEvaluator;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.util.ObjectUtil;

/**
 * This class adapts the unconstrained nonlinear minimization algorithms in
 * the "minimization" package to the task of estimating locally optimal
 * (minimum-cost) parameter sets.  This allows us to use algorithms like BFGS
 * {@code FunctionMinimizerBFGS} to find locally optimal parameters of, for
 * example, a {@code DifferentiableFeedforwardNeuralNetwork}.  Any first-order
 * derivative {@code FunctionMinimizer} may be dropped into this class.
 * <BR><BR>
 * My current preference is for using BFGS ({@code FunctionMinimizerBFGS}) to
 * solve virtually all problems.  However, when there are too many parameters,
 * then Liu-Storey conjugate gradient ({@code FunctionMinimizerLiuStorey}) is
 * another good choice.
 * <BR><BR>
 * When first-order derivative information is not available, then you may use
 * either automatic differentiation ({@code GradientDescendableApproximator})
 * or the derivative-free minimization routines, such as those used by
 * {@code ParameterDerivativeFreeCostMinimizer}.
 * 
 * @see gov.sandia.cognition.learning.algorithm.minimization.FunctionMinimizer
 * @see gov.sandia.cognition.learning.algorithm.regression.ParameterDerivativeFreeCostMinimizer
 * @author Kevin R. Dixon
 * @since 2.1
 */
public class ParameterDifferentiableCostMinimizer
    extends AbstractMinimizerBasedParameterCostMinimizer
        <GradientDescendable,DifferentiableEvaluator<Vector,Double,Vector>>
{
    
    /**
     * Default function minimizer,
     * FunctionMinimizerBFGS with LineMinimizerBacktracking
     */
    public static final FunctionMinimizer<Vector,Double,DifferentiableEvaluator<? super Vector,Double,Vector>> DEFAULT_FUNCTION_MINIMIZER =
        new FunctionMinimizerBFGS( new LineMinimizerBacktracking() );
    
    /** 
     * Creates a new instance of ParameterDifferentiableCostMinimizer 
     */
    public ParameterDifferentiableCostMinimizer()
    {
        this( ObjectUtil.cloneSafe( DEFAULT_FUNCTION_MINIMIZER ) );
    }

    /**
     * Creates a new instance of ParameterDerivativeFreeCostMinimizer 
     * @param minimizer
     * Minimization algorithm used to find locally optimal parameters
     */
    public ParameterDifferentiableCostMinimizer(
        FunctionMinimizer<Vector,Double,? super DifferentiableEvaluator<Vector,Double,Vector>> minimizer )
    {
        super( minimizer );
    }

    @Override
    public ParameterDifferentiableCostMinimizer clone()
    {
        return (ParameterDifferentiableCostMinimizer) super.clone();
    }

    @SuppressWarnings("unchecked")
    @Override
    public ParameterCostEvaluatorDerivativeBased createInternalFunction()
    {
        return new ParameterCostEvaluatorDerivativeBased( 
            this.getResult(), (DifferentiableCostFunction<Vector, Vector, ? super GradientDescendable>) this.getCostFunction());
    }
    
    /**
     * Function that maps the parameters of an object to its inputs, so that
     * minimization algorithms can tune the parameters of an object against
     * a cost function.  Uses algebraic derivatives.
     */
    public static class ParameterCostEvaluatorDerivativeBased
        extends AbstractScalarFunction<Vector>
        implements DifferentiableEvaluator<Vector,Double,Vector>
    {
        
        /**
         * Object that we're tweaking the parameters of.
         */
        private GradientDescendable internalFunction;
        
        /**
         * Cost function against which to evaluate the cost of the object.
         */
        private DifferentiableCostFunction<Vector, Vector, ? super GradientDescendable> costFunction;
        
        /**
         * Creates a new instance of ParameterCostEvaluatorDerivativeBased
         * @param internalFunction
         * Object that we're tweaking the parameters of.
         * @param costFunction
         * Cost function against which to evaluate the cost of the object.
         */
        public ParameterCostEvaluatorDerivativeBased(
            GradientDescendable internalFunction,
            DifferentiableCostFunction<Vector, Vector, ? super GradientDescendable> costFunction )
        {
            this.internalFunction = internalFunction;
            this.costFunction = costFunction;
        }

        @Override
        public Vector differentiate(
            Vector input )
        {
            this.internalFunction.convertFromVector( input );
            return this.costFunction.computeParameterGradient( internalFunction );
        }

        @Override
        public double evaluateAsDouble(
            Vector input )
        {
            this.internalFunction.convertFromVector( input );
            return this.costFunction.computeCost(this.internalFunction);
        }

        @Override
        public ParameterCostEvaluatorDerivativeBased clone()
        {
            ParameterCostEvaluatorDerivativeBased clone =
                (ParameterCostEvaluatorDerivativeBased) super.clone();
            clone.costFunction = ObjectUtil.cloneSmart( this.costFunction );
            clone.internalFunction = ObjectUtil.cloneSafe( this.internalFunction );
            return clone;
        }
        
    }

}
