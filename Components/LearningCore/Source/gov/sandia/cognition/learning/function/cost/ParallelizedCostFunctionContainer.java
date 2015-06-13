/*
 * File:                ParallelizedCostFunctionContainer.java
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

import gov.sandia.cognition.algorithm.ParallelAlgorithm;
import gov.sandia.cognition.algorithm.ParallelUtil;
import gov.sandia.cognition.evaluator.Evaluator;
import gov.sandia.cognition.learning.data.InputOutputPair;
import gov.sandia.cognition.learning.data.SequentialDataMultiPartitioner;
import gov.sandia.cognition.learning.data.TargetEstimatePair;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.util.ObjectUtil;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A cost function that automatically splits a ParallelizableCostFunction
 * across multiple cores/processors to speed up computation.
 * @author Kevin R. Dixon
 * @since 2.1
 * @param <InputType> The type of input for the evaluated function.
 * @param <OutputType> The type of output for the evaluated function.
 * @param <EvaluatedType> The type of evaluated function.
 * @param <DifferentiableEvaluatedType> The type of evaluated function that
 *      can be differentiated.
 */
public class ParallelizedCostFunctionContainer<InputType, OutputType, EvaluatedType extends Evaluator<? super InputType, ? extends OutputType>, DifferentiableEvaluatedType extends EvaluatedType>
    extends AbstractSupervisedCostFunction<InputType, OutputType, EvaluatedType>
    implements DifferentiableCostFunction<InputType, OutputType, DifferentiableEvaluatedType>,
        ParallelAlgorithm
{
    
    /**
     * Cost function to parallelize
     */
    private ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction;

    /**
     * Collection of evaluation thread calls
     */
    private transient ArrayList<SubCostEvaluate> evaluationComponents;
    
    /**
     * Collection of evaluation gradient calls
     */
    private transient ArrayList<SubCostGradient> gradientComponents;
    
    /**
     * Thread pool used to parallelize the computation
     */
    private transient ThreadPoolExecutor threadPool;

    /**
     * Default constructor for ParallelizedCostFunctionContainer.
     */
    public ParallelizedCostFunctionContainer()
    {
        this(null);
    }
    
    /**
     * Creates a new instance of ParallelizedCostFunctionContainer
     * @param costFunction
     * Cost function to parallelize
     */
    public ParallelizedCostFunctionContainer(
        ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction )
    {
        this( costFunction, ParallelUtil.createThreadPool() );
    }
    
    /**
     * Creates a new instance of ParallelizedCostFunctionContainer
     * @param threadPool 
     * Thread pool used to parallelize the computation
     * @param costFunction
     * Cost function to parallelize
     */
    public ParallelizedCostFunctionContainer(
        ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction,
        ThreadPoolExecutor threadPool )
    {
        this.setCostFunction( costFunction );
        this.setThreadPool( threadPool );
    }       
    
    @Override
    public ParallelizedCostFunctionContainer<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> clone()
    {
        @SuppressWarnings("unchecked")
        ParallelizedCostFunctionContainer<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> clone =
            (ParallelizedCostFunctionContainer<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType>) super.clone();
        clone.setCostFunction( ObjectUtil.cloneSmart( this.getCostFunction() ) );
        clone.setThreadPool(
            ParallelUtil.createThreadPool( this.getNumThreads() ) );
        return clone;
    }    
    
    /**
     * Getter for costFunction
     * @return
     * Cost function to parallelize
     */
    public ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> getCostFunction()
    {
        return this.costFunction;
    }
    
    /**
     * Setter for costFunction
     * @param costFunction
     * Cost function to parallelize
     */
    public void setCostFunction(
        ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction )
    {
        this.costFunction = costFunction;
        this.evaluationComponents = null;
        this.gradientComponents = null;
    }
    
    /**
     * Splits the data across the numComponents cost functions
     */
    protected void createPartitions()
    {
        int numThreads = this.getNumThreads();
        ArrayList<ArrayList<InputOutputPair<? extends InputType, OutputType>>> partitions =
            SequentialDataMultiPartitioner.create(
                this.getCostParameters(), numThreads );
        this.evaluationComponents = new ArrayList<SubCostEvaluate>( numThreads );
        this.gradientComponents = new ArrayList<SubCostGradient>( numThreads );
        for( int i = 0; i < numThreads; i++ )
        {
            ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> subcost =
                ObjectUtil.cloneSmart(this.getCostFunction());
            subcost.setCostParameters( partitions.get(i) );
            this.evaluationComponents.add( new SubCostEvaluate( subcost, null ) );
            this.gradientComponents.add( new SubCostGradient( subcost, null ) );
        }
        
    }

    @Override
    public void setCostParameters(
        Collection<? extends InputOutputPair<? extends InputType, OutputType>> costParameters )
    {
        super.setCostParameters( costParameters );
        this.evaluationComponents = null;
        this.gradientComponents = null;
    }
    
    @Override
    public double evaluateAsDouble(
        EvaluatedType evaluator )
    {
        
        if( this.evaluationComponents == null )
        {
            this.createPartitions();
        }
        
        // Set the subtasks
        for( SubCostEvaluate sce : this.evaluationComponents )
        {
            sce.evaluator = evaluator;
        }
        
        Collection<Object> partialResults = null;
        try
        {
            partialResults = ParallelUtil.executeInParallel(
                this.evaluationComponents, this.getThreadPool() );
        }
        catch (Exception ex)
        {
            Logger.getLogger( ParallelizedCostFunctionContainer.class.getName() ).log( Level.SEVERE, null, ex );
        }
        
        return this.getCostFunction().evaluateAmalgamate( partialResults );
        
    }
        
    @Override
    public double evaluatePerformanceAsDouble(
        Collection<? extends TargetEstimatePair<? extends OutputType, ? extends OutputType>> data )
    {
        return this.getCostFunction().evaluatePerformanceAsDouble( data );
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
        
        if (this.gradientComponents == null)
        {
            this.createPartitions();
        }

        // Create the subtasks
        for (SubCostGradient eval : this.gradientComponents)
        {
            eval.evaluator = function;
        }

        Collection<Object> results = null;
        try
        {
            results = ParallelUtil.executeInParallel(
                this.gradientComponents, this.getThreadPool() );
        }
        catch (Exception ex)
        {
            Logger.getLogger( ParallelizedCostFunctionContainer.class.getName() ).log( Level.SEVERE, null, ex );
        }
        
        return this.getCostFunction().computeParameterGradientAmalgamate( results );
        
    }

    public ThreadPoolExecutor getThreadPool()
    {
        if( this.threadPool == null )
        {
            this.setThreadPool( ParallelUtil.createThreadPool() );
        }
        
        return this.threadPool;
    }

    public void setThreadPool(
        ThreadPoolExecutor threadPool )
    {
        this.threadPool = threadPool;
    }

    public int getNumThreads()
    {
        return ParallelUtil.getNumThreads( this );
    }
    
    /**
     * Creates the thread pool using the Foundry's global thread pool.
     */
    protected void createThreadPool()
    {
        this.setThreadPool( ParallelUtil.createThreadPool() );
    }

    /**
     * Callable task for the evaluate() method.
     */
    protected class SubCostEvaluate
        implements Callable<Object>
    {
        
        /**
         * Parallel cost function
         */
        private ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction;
        
        /**
         * Evaluator for which to compute the cost
         */
        private EvaluatedType evaluator;
        
        /**
         * Creates a new instance of SubCostEvaluate
         * @param costFunction
         * Parallel cost function
         * @param evaluator
         * Evaluator for which to compute the cost
         */
        public SubCostEvaluate(
            ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction,
            EvaluatedType evaluator )
        {
            this.costFunction = costFunction;
            this.evaluator = evaluator;
        }

        public Object call()
        {
            return this.costFunction.evaluatePartial( this.evaluator );
        }
        
    }
    
    /**
     * Callable task for the computeGradient() method
     */
    protected class SubCostGradient
        implements Callable<Object>
    {
        
        /**
         * Parallel cost function
         */
        private ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction;
        
        /**
         * Function for which to compute the gradient
         */
        private DifferentiableEvaluatedType evaluator;
        
        /**
         * Creates a new instance of SubCostGradient
         * @param costFunction
         * Parallel cost function
         * @param evaluator
         * Function for which to compute the gradient
         */
        public SubCostGradient(
            ParallelizableCostFunction<InputType, OutputType, EvaluatedType, DifferentiableEvaluatedType> costFunction,
            DifferentiableEvaluatedType evaluator )
        {
            this.costFunction = costFunction;
            this.evaluator = evaluator;
        }

        public Object call()
        {
            return this.costFunction.computeParameterGradientPartial( this.evaluator );
        }
        
    }

}
