/*
 * File:            VectorizableEvaluator.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2015 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.math.matrix;

import gov.sandia.cognition.evaluator.Evaluator;

/**
 * Interface for an {@link Evaluator} that is also {@link Vectorizable} in
 * terms of its parameters. This acts as a common conjunction interface.
 * 
 * @param   <InputType> The type of input.
 * @param   <OutputType> The type of output.
 * @author  Justin Basilico
 * @since   3.4.2
 */
public interface VectorizableEvaluator<InputType, OutputType>
    extends Evaluator<InputType, OutputType>, Vectorizable
{
    
}
