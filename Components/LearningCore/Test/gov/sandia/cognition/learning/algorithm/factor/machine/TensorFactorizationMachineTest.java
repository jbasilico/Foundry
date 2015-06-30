/*
 * File:            TensorFactorizationMachineTest.java
 * Authors:         Justin Basilico
 * Project:         Cognitive Foundry
 * 
 * Copyright 2014 Cognitive Foundry. All rights reserved.
 */

package gov.sandia.cognition.learning.algorithm.factor.machine;

import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.Vector1;
import gov.sandia.cognition.math.matrix.mtj.Vector3;
import gov.sandia.cognition.util.ArgumentChecker;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import static org.junit.Assert.*;
import org.junit.Test;

/**
 * Unit tests for {@link TensorFactorizationMachine}.
 * 
 * @author  Justin Basilico
 * @since   3.4.0
 */
public class TensorFactorizationMachineTest
{
    
    protected Random random = new Random(4444);
    protected double epsilon = 1e-5;
    
    protected VectorFactory<?> vfs = VectorFactory.getSparseDefault();
    protected VectorFactory<?> vfd = VectorFactory.getDenseDefault();
    protected MatrixFactory<?> mf = MatrixFactory.getDenseDefault();
    
    /**
     * Creates a new test.
     */
    public TensorFactorizationMachineTest()
    {
        super();
    }
    
    /**
     * Test of constants of class TensorFactorizationMachine.
     */
    @Test
    public void testConstants()
    {
        int ways = 10 + 1;
        assertEquals(ways, TensorFactorizationMachine.PARTITIONS_PER_WAY.length);
        assertEquals(ways, TensorFactorizationMachine.COEFFICIENTS_PER_WAY.length);
        assertEquals(ways, TensorFactorizationMachine.FACTORIAL_PER_WAY.length);
        
        assertEquals(0, TensorFactorizationMachine.PARTITIONS_PER_WAY[0].length);
        assertEquals(0, TensorFactorizationMachine.COEFFICIENTS_PER_WAY[0].length);
        assertEquals(1, TensorFactorizationMachine.FACTORIAL_PER_WAY[0]);
        
        int factorial = 1;
        for (int way = 1; way < ways; way++)
        {
            factorial *= way;
            
            int[][] partitions = TensorFactorizationMachine.PARTITIONS_PER_WAY[way];
            int[] coefficients = TensorFactorizationMachine.COEFFICIENTS_PER_WAY[way];
            int termCount = partitions.length;
            assertEquals(termCount, coefficients.length);
            
            for (int[] partition : partitions)
            {
                int sum = 0;
                for (int part : partition)
                {
                    assertTrue(part > 0);
                    sum += part;
                }
                assertEquals(way, sum);
            }
            
            int coefficientSum = 0;
            for (int coefficient : coefficients)
            {
                coefficientSum += coefficient;
            }
            if (way > 1)
            {
                assertEquals(0, coefficientSum);
            }
            
            assertEquals(factorial, TensorFactorizationMachine.FACTORIAL_PER_WAY[way]);
        }
    }
    
    /**
     * Test of constructors, of class TensorFactorizationMachine.
     */
    @Test
    public void testConstructors()
    {
        double bias = 0.0;
        Vector weights = null;
        Matrix[] factorsPerWay = new Matrix[0];
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(bias, instance.getBias(), 0.0);
        assertSame(weights, instance.getWeights());
        assertArrayEquals(factorsPerWay, instance.getFactorsPerWay());
        
        instance = new TensorFactorizationMachine(12, 5, 4);
        assertEquals(bias, instance.getBias(), 0.0);
        assertEquals(12, instance.getWeights().getDimensionality());
        assertEquals(5, instance.getFactors(2).getNumRows());
        assertEquals(12, instance.getFactors(2).getNumColumns());
        assertEquals(4, instance.getFactors(3).getNumRows());
        assertEquals(12, instance.getFactors(3).getNumColumns());
        assertEquals(0.0, instance.getWeights().sum(), 0.0);
        assertEquals(0.0, instance.getFactors(2).sumOfRows().sum(), 0.0);
        assertEquals(0.0, instance.getFactors(3).sumOfRows().sum(), 0.0);
        
        bias = 0.4;
        weights = VectorFactory.getSparseDefault().createVector(11);
        Matrix factors2 = MatrixFactory.getSparseDefault().createMatrix(6, 11);
        Matrix factors3 = MatrixFactory.getSparseDefault().createMatrix(4, 11);
        factorsPerWay = new Matrix[] { factors2, factors3 };
        instance = new TensorFactorizationMachine(bias, weights, factorsPerWay);
        assertEquals(bias, instance.getBias(), 0.0);
        assertSame(weights, instance.getWeights());
        assertSame(factorsPerWay, instance.getFactorsPerWay());
    }

    /**
     * Test of clone method, of class TensorFactorizationMachine.
     */
    @Test
    public void testClone()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        TensorFactorizationMachine clone = instance.clone();
       
        assertNotSame(instance, clone);
        assertNotNull(clone);
        assertNotSame(clone, instance.clone());
        assertEquals(instance.getBias(), clone.getBias(), 0.0);
        assertNull(clone.getWeights());
        assertEquals(0, clone.getFactorsPerWay().length);
        
        instance.setBias(3.4);
        instance.setWeights(VectorFactory.getDefault().createUniformRandom(
            5, -1, 1, random));
        instance.setFactorsPerWay(
            MatrixFactory.getDefault().createUniformRandom(3, 5, -1, 1, random),
            null,
            MatrixFactory.getDefault().createUniformRandom(2, 5, -1, 1, random));
        clone = instance.clone();
        
        assertNotSame(instance, clone);
        assertNotNull(clone);
        assertNotSame(clone, instance.clone());
        assertEquals(instance.getBias(), clone.getBias(), 0.0);
        assertEquals(instance.getWeights(), clone.getWeights());
        assertEquals(instance.getFactors(2), clone.getFactors(2));
        assertEquals(instance.getFactors(3), clone.getFactors(3));
        assertEquals(instance.getFactors(4), clone.getFactors(4));
        assertNotSame(instance.getWeights(), clone.getWeights());
        assertNotSame(instance.getFactors(2), clone.getFactors(2));
        assertNull(clone.getFactors(3));
        assertNotSame(instance.getFactors(4), clone.getFactors(4));
    }

    /**
     * Test of evaluateAsDouble method, of class TensorFactorizationMachine.
     */
    @Test
    public void testEvaluateAsDouble()
    {
        int d = 10; //3 + this.random.nextInt(10);
        int k = 5; // 1 + this.random.nextInt(d - 1);
        Vector x1 = vfd.createUniformRandom(d, -10, 10, random);
        Vector x2 = vfd.createUniformRandom(d, -10, 10, random);
        List<Vector> xs = Arrays.asList(x1, x2);
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(0.0, instance.evaluateAsDouble(null), 0.0);
        
        instance = new TensorFactorizationMachine(d, 0, 0, 0, 0);
        double b = this.random.nextGaussian();
        instance.setBias(b);
        assertEquals(b, instance.evaluateAsDouble(x1), 0.0);
        assertEquals(b, instance.evaluateAsDouble(x2), 0.0);
        
        
        Vector w = vfd.createVector(d);
        instance.setWeights(w);
        assertEquals(b, instance.evaluateAsDouble(x1), 0.0);
        assertEquals(b, instance.evaluateAsDouble(x2), 0.0);
        
        w = vfd.createUniformRandom(d, -1, 1, random);
        instance.setWeights(w);
        assertEquals(b + w.dotProduct(x1), instance.evaluateAsDouble(x1), 0.0);
        assertEquals(b + w.dotProduct(x2), instance.evaluateAsDouble(x2), 0.0);
        
        Matrix v = mf.createMatrix(k, d);
        instance.setFactors(2, v);
        assertEquals(b + w.dotProduct(x1), instance.evaluateAsDouble(x1), 0.0);
        assertEquals(b + w.dotProduct(x2), instance.evaluateAsDouble(x2), 0.0);
        
        v = mf.createUniformRandom(k, d, -1, 1, random);
        instance.setFactors(2, v);
        
        // This is the way in the base formula to compute it that is O(kn^2)
        // rather than the way it is really computed as O(kn).
        for (Vector x : xs)
        {
            double expected = b + w.dotProduct(x);
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    expected += x.getElement(i) * x.getElement(j)
                        * v.getColumn(i).dotTimes(v.getColumn(j)).sum();
                }
            }
            assertEquals(expected, instance.evaluateAsDouble(x), epsilon);
        }
        
        // Should also be the same as a standard FactorizationMachine.
        {
            FactorizationMachine other = new FactorizationMachine(
                instance.getBias(), instance.getWeights().clone(), 
                instance.getFactors(2).clone());
            
            for (Vector x : xs)
            {
                assertEquals(other.evaluateAsDouble(x), instance.evaluateAsDouble(x), epsilon);
            }
        }
        
        // 3-way interactions.
        instance.setFactors(2, null);
        instance.setFactors(3, v);
        
        // This is the way in the base formula to compute it that is O(kn^3)
        // rather than the way it is really computed as O(kn).
        for (Vector x : xs)
        {
            double expected = b + w.dotProduct(x);
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    for (int m = j + 1; m < d; m++)
                    {
                        double value = x.getElement(i) * x.getElement(j) * x.getElement(m)
                            * v.getColumn(i).dotTimes(v.getColumn(j)).dotTimes(v.getColumn(m)).sum();
                        expected += value;
                    }
                }
            }

            assertEquals(expected, instance.evaluateAsDouble(x), epsilon);
            assertEquals(expected, b + w.dotProduct(x) + computePairwise(x, v, 3), epsilon);
        }
        
        // 4-way interactions only.
        instance.setFactors(2, null);
        instance.setFactors(3, null);
        instance.setFactors(4, v);
        
        // This is the way in the base formula to compute it that is O(kn^3)
        // rather than the way it is really computed as O(kn).
        for (Vector x : Arrays.asList(x1, x2))
        {
            double expected = b + w.dotProduct(x);
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    for (int m = j + 1; m < d; m++)
                    {
                        for (int n = m + 1; n < d; n++)
                        {
                            expected += x.getElement(i) * x.getElement(j) * x.getElement(m) * x.getElement(n)
                                * v.getColumn(i).dotTimes(v.getColumn(j)).dotTimes(v.getColumn(m)).dotTimes(v.getColumn(n)).sum();
                        }
                    }
                }
            }
            
            assertEquals(expected, instance.evaluateAsDouble(x), epsilon);
            assertEquals(expected, b + w.dotProduct(x) + computePairwise(x, v, 4), epsilon);
        }
        
        
        fail("The test case is a prototype.");
    }
    
    @Test
    public void testEvaluateAsDoubleForHigherWays()
    {
        int k = 12;
        int d = 14;
        int n = 20;
        for (int way = 2; way <= 10; way++)
        {
            System.out.println("Computing coefficients for way " + way);
            Matrix v = mf.createUniformRandom(k, d, -1, 1, random);
            TensorFactorizationMachine instance = new TensorFactorizationMachine();
            instance.setFactorsPerWay(new Matrix[way - 1]);
            instance.setFactors(way, v);
            for (int example = 0; example < n; example++)
            {
                Vector x = vfd.createUniformRandom(d, -10, 10, random);
                double expected = computePairwise(x, v, way);
                assertEquals(expected, instance.evaluateAsDouble(x), epsilon);
            }
        }
    }
    
    @Test
    public void testComputeCoefficients()
    {

        for (int way = 5; way <= 10; way++)
        {
            System.out.println("Computing coefficients for way " + way);
            int k = way + 2; // 1 + this.random.nextInt(d - 1);
            int d = k + 2; //3 + this.random.nextInt(10);
            int n = 2 * way;


            Matrix v = MatrixFactory.getDenseDefault().createUniformRandom(k, d, -1, 1, random);
            System.out.println("Solving for way = " + way);


            int[][] partitions = createPartitions(way);
            Matrix coeffs = MatrixFactory.getDenseDefault().createMatrix(n * k, partitions.length);
            Vector targets = VectorFactory.getDenseDefault().createVector(n * k);
            for (int example = 0; example < n; example++)
            {
                Vector x = VectorFactory.getDefault().createUniformRandom(d, -10, 10, random);
                System.out.println("Expected: " + computePairwise(x, v, way));
                for (int factor = 0; factor < k; factor++)
                {
                    double[] powers = new double[way + 1];
                    Vector vf = v.getRow(factor);
                    for (int i = 0; i < d; i++)
                    {
                        double value = x.get(i) * vf.get(i);

                        double product = 1.0;
                        for (int j = 1; j <= way; j++)
                        {
                            product *= value;
                            powers[j] += product;
                        }
                    }

                    double expected = computePairwise(x, MatrixFactory.getDenseDefault().copyRowVectors(vf), way);

                    double[] features = new double[partitions.length];
                    for (int i = 0; i < partitions.length; i++)
                    {
                        int[] partition = partitions[i];
                        double value = 1.0;
                        for (int part : partition)
                        {
                            value *= powers[part];
                        }
                        features[i] = value;
                    }

                    coeffs.setRow(k * example + factor, VectorFactory.getDenseDefault().copyValues(features));
                    targets.set(k * example + factor, expected);
                    System.out.println("f: " + factor + ": " + Arrays.toString(features) + " -> " + expected);
                }
            }

            int factorial = 1;
            for (int i = 2; i <= way; i++)
            {
                factorial *= i;
            }

            System.out.println("All : " + coeffs.sumOfRows() + " -> " + targets.sum());

            Vector solved = coeffs.solve(targets);
            System.out.println(solved);
            double first = solved.get(0);
            for (int i = 0; i < partitions.length; i++)
            {
                System.out.printf("%d: %.1f", i, solved.get(i) / first);
                System.out.println(" " + Arrays.toString(partitions[i]));
            }
            for (int i = 0; i < partitions.length; i++)
            {
                System.out.printf("%.1f, ", solved.get(i) / first);
            }
            System.out.println();
            System.out.println(solved.scale(factorial));
            System.out.println(coeffs.times(solved));
            System.out.println(targets.minus(coeffs.times(solved)).norm2());
        }
    }

    /**
     * Test of computeParameterGradient method, of class TensorFactorizationMachine.
     */
    @Test
    public void testComputeParameterGradient()
    {
        VectorFactory<?> vf = VectorFactory.getSparseDefault();
        MatrixFactory<?> mf = MatrixFactory.getDenseDefault();
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        Vector input = vf.createVector(0);
        Vector result = instance.computeParameterGradient(input);
        assertEquals(1, result.getDimensionality());
        assertEquals(1.0, result.getElement(0), 0.0);
        
        int d = 10;
        instance.setWeights(VectorFactory.getDenseDefault().createVector(d));
        input = vf.createUniformRandom(d, -10, 10, random);
        result = instance.computeParameterGradient(input);
        assertEquals(1 + d, result.getDimensionality());
        assertEquals(1.0, result.getElement(0), 0.0);
        assertEquals(input, result.subVector(1, d));
        
        int k = 4;
        // Test that 2-way interactions match other FactorizationMachine class.
        {
            Matrix factors2 = mf.createUniformRandom(k, d, -1, 1, random);
            instance.setFactorsPerWay(factors2.clone());
            input = vf.createUniformRandom(d, -10, 10, random);
            result = instance.computeParameterGradient(input);
            Vector expected = new FactorizationMachine(
                instance.getBias(), instance.getWeights().clone(),
                instance.getFactors(2).clone()).computeParameterGradient(input);
            assertTrue(expected.equals(result, epsilon));
        }
        
        for (int way = 2; way <= 6; way++)
        {
            instance.setFactorsPerWay(new Matrix[5]);
            instance.setFactors(way, mf.createUniformRandom(k, d, -10, 10, random));
            input = vf.createUniformRandom(d, -10, 10, random);
            result = instance.computeParameterGradient(input);
            assertEquals(1 + d + k * d, result.getDimensionality());
            assertEquals(1.0, result.getElement(0), 0.0);
            assertEquals(input, result.subVector(1, d));
            Vector factorGradients = result.subVector(d + 1, d + d * k);
            Matrix expectedGradients = computePairwiseGradients(
                    input, instance.getFactors(way), way);
            Vector expected = expectedGradients.transpose().convertToVector();
            assertTrue(expected.equals(factorGradients, epsilon));
        }
        
        // Test stacking multiple ways in the gradient.
        {
            int k2 = 4;
            int k4 = 6;
            Matrix factors2 = mf.createUniformRandom(k2, d, -1, 1, random);
            Matrix factors4 = mf.createUniformRandom(k4, d, -1, 1, random);
            instance.setFactorsPerWay(factors2.clone(), null, factors4.clone());
            input = vf.createUniformRandom(d, -10, 10, random);
            result = instance.computeParameterGradient(input);
            assertEquals(1 + d + k2 * d + k4 * d, result.getDimensionality());
            assertEquals(1.0, result.getElement(0), 0.0);
            assertEquals(input, result.subVector(1, d));
            Vector factorGradients = result.subVector(d + 1, d + d * k2 + d * k4);
            Matrix expectedGradients2 = computePairwiseGradients(
                input, instance.getFactors(2), 2);
            Matrix expectedGradients4 = computePairwiseGradients(
                input, instance.getFactors(4), 4);
            Vector expected = expectedGradients2.transpose().convertToVector().stack(
                expectedGradients4.transpose().convertToVector());
            assertTrue(expected.equals(factorGradients, epsilon));
        }
    }
    
    /**
     * Test of convertToVector method, of class TensorFactorizationMachine.
     */
    @Test
    public void testConvertToVector()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        Vector result = instance.convertToVector();
        assertEquals(instance.getParameterCount(), result.getDimensionality());
        assertTrue(result.isZero());
        
        int d = 7;
        int k2 = 4;
        int k4 = 3;
        instance = new TensorFactorizationMachine(d, k2, 0, k4);
        result = instance.convertToVector();
        assertEquals(instance.getParameterCount(), result.getDimensionality());
        assertTrue(result.isZero());
        
        double bias = this.random.nextGaussian();
        Vector weights = VectorFactory.getDefault().createUniformRandom(d, -1, 1, random);
        Matrix factors2 = MatrixFactory.getDefault().createUniformRandom(k2, d, -1, 1, random);
        Matrix factors4 = MatrixFactory.getDefault().createUniformRandom(k4, d, -1, 1, random);
        instance = new TensorFactorizationMachine(bias, weights.clone(), factors2.clone(), null, factors4.clone());
        result = instance.convertToVector();
        assertEquals(instance.getParameterCount(), result.getDimensionality());
        assertTrue(result.equals(new Vector1(bias).stack(weights).stack(factors2.transpose().convertToVector()).stack(factors4.transpose().convertToVector())));
        
        // Try with weights disabled.
        instance.setWeights(null);
        result = instance.convertToVector();
        assertEquals(instance.getParameterCount(), result.getDimensionality());
        assertTrue(result.equals(new Vector1(bias).stack(factors2.transpose().convertToVector()).stack(factors4.transpose().convertToVector())));
        
        // Try with factors disabled.
        instance.setWeights(weights.clone());
        instance.setFactorsPerWay();
        result = instance.convertToVector();
        assertEquals(instance.getParameterCount(), result.getDimensionality());
        assertTrue(result.equals(new Vector1(bias).stack(weights)));
    }
    
    /**
     * Test of convertFromVector method, of class TensorFactorizationMachine.
     */
    @Test
    public void testConvertFromVector()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        Vector converted = instance.convertToVector();
        Vector expected = converted.clone();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        
        int d = 7;
        int k2 = 4;
        int k4 = 3;
        instance = new TensorFactorizationMachine(d, k2, 0, k4);
        converted = instance.convertToVector();
        expected = converted.clone();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        
        double bias = this.random.nextGaussian();
        Vector weights = VectorFactory.getDefault().createUniformRandom(d, -1, 1, random);
        Matrix factors2 = MatrixFactory.getDefault().createUniformRandom(k2, d, -1, 1, random);
        Matrix factors4 = MatrixFactory.getDefault().createUniformRandom(k4, d, -1, 1, random);
        instance = new TensorFactorizationMachine(bias, weights.clone(), factors2.clone(), null, factors4.clone());
        converted = instance.convertToVector();
        expected = converted.clone();
        instance.convertFromVector(converted);
        assertEquals(expected, instance.convertToVector());
        assertEquals(bias, instance.getBias(), 0.0);
        assertEquals(weights, instance.getWeights());
        assertEquals(factors2, instance.getFactors(2));
        assertNull(instance.getFactors(3));
        assertEquals(factors4, instance.getFactors(4));
        
        instance = new TensorFactorizationMachine(d, k2, 0, k4);
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        assertEquals(bias, instance.getBias(), 0.0);
        assertEquals(weights, instance.getWeights());
        assertEquals(factors2, instance.getFactors(2));
        assertNull(instance.getFactors(3));
        assertEquals(factors4, instance.getFactors(4));
        
        // Try with weights disabled.
        instance.setWeights(null);
        converted = instance.convertToVector();
        expected = converted.clone();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        instance.setBias(0.0);
        instance.getFactors(2).zero();
        instance.getFactors(4).zero();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        assertEquals(bias, instance.getBias(), 0.0);
        assertNull(instance.getWeights());
        assertEquals(factors2, instance.getFactors(2));
        assertNull(instance.getFactors(3));
        assertEquals(factors4, instance.getFactors(4));
        
        // Try with factors disabled.
        instance.setWeights(weights.clone());
        instance.setFactorsPerWay(new Matrix[3]);
        converted = instance.convertToVector();
        expected = converted.clone();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        instance.setBias(0.0);
        instance.getWeights().zero();
        instance.convertFromVector(converted);
        assertTrue(expected.equals(instance.convertToVector()));
        assertEquals(bias, instance.getBias(), 0.0);
        assertEquals(weights, instance.getWeights());
        assertNull(instance.getFactors(2));
        assertNull(instance.getFactors(3));
        assertNull(instance.getFactors(4));
    }
    
    /**
     * Test of getParameterCount method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetParameterCount()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(1, instance.getParameterCount());
        
        instance.setFactorsPerWay(
            MatrixFactory.getDefault().createMatrix(12, 4),
            null,
            MatrixFactory.getDefault().createMatrix(14, 4));
        assertEquals(1 + 12 * 4 + 14 * 4, instance.getParameterCount());
        
        instance.setFactorsPerWay();
        instance.setWeights(VectorFactory.getDefault().createVector(4));
        assertEquals(1 + 4, instance.getParameterCount());
        
        instance = new TensorFactorizationMachine(40, 7, 0, 8);
        assertEquals(1 + 40 + 40 * 7 + 40 * 8, instance.getParameterCount());
    }
    
    /**
     * Test of getInputDimensionality method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetInputDimensionality()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(0, instance.getInputDimensionality());
        
        instance.setWeights(new Vector3());
        assertEquals(3, instance.getInputDimensionality());
        
        instance.setWeights(null);
        instance.setFactorsPerWay(null,
            MatrixFactory.getDefault().createMatrix(12, 40),
            MatrixFactory.getDefault().createMatrix(12, 40));
        assertEquals(40, instance.getInputDimensionality());
        
        instance = new TensorFactorizationMachine(40, 7, 0, 7);
        assertEquals(40, instance.getInputDimensionality());
    }

    /**
     * Test of getMaxWay method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetMaxWay()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(1, instance.getMaxWay());
        
        instance.setFactorsPerWay(new Matrix[1]);
        assertEquals(2, instance.getMaxWay());

        instance.setFactorsPerWay(new Matrix[2]);
        assertEquals(3, instance.getMaxWay());

        instance.setFactorsPerWay(new Matrix[3]);
        assertEquals(4, instance.getMaxWay());
    }

    /**
     * Test of getFactors method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetFactors()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertNull(instance.getFactors(2));
        assertNull(instance.getFactors(3));
        assertNull(instance.getFactors(4));
        assertNull(instance.getFactors(5));
        
        Matrix factors2 = MatrixFactory.getDefault().createMatrix(12, 4);
        Matrix factors3 = null;
        Matrix factors4 = MatrixFactory.getDefault().createMatrix(14, 4);
        
        instance.setFactorsPerWay(factors2, factors3, factors4);
        assertSame(factors2, instance.getFactors(2));
        assertSame(factors3, instance.getFactors(3));
        assertSame(factors4, instance.getFactors(4));
        assertNull(instance.getFactors(5));

        instance = new TensorFactorizationMachine(40, 7, 0, 14);
        assertEquals(7, instance.getFactors(2).getNumRows());
        assertNull(instance.getFactors(3));
        assertEquals(14, instance.getFactors(4).getNumRows());
        assertNull(instance.getFactors(5));
        
        int[] badWays = {0, 1, -1};
        for (int badWay : badWays)
        {
            boolean exceptionThrown = false;
            try
            {
                instance.getFactors(badWay);
            }
            catch (IllegalArgumentException e)
            {
                exceptionThrown = true;
            }
            finally
            {
                assertTrue(exceptionThrown);
            }
        }
    }

    /**
     * Test of getFactorCount method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetFactorCount()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(0, instance.getFactorCount(0));
        assertEquals(0, instance.getFactorCount(1));
        assertEquals(0, instance.getFactorCount(2));
        assertEquals(0, instance.getFactorCount(3));
        assertEquals(0, instance.getFactorCount(4));
        assertEquals(0, instance.getFactorCount(5));
        
        instance.setFactorsPerWay(
            MatrixFactory.getDefault().createMatrix(12, 4),
            null,
            MatrixFactory.getDefault().createMatrix(14, 4));
        assertEquals(0, instance.getFactorCount(0));
        assertEquals(0, instance.getFactorCount(1));
        assertEquals(12, instance.getFactorCount(2));
        assertEquals(0, instance.getFactorCount(3));
        assertEquals(14, instance.getFactorCount(4));
        assertEquals(0, instance.getFactorCount(5));
        
        instance = new TensorFactorizationMachine(40, 7, 0, 14);
        assertEquals(0, instance.getFactorCount(0));
        assertEquals(0, instance.getFactorCount(1));
        assertEquals(7, instance.getFactorCount(2));
        assertEquals(0, instance.getFactorCount(3));
        assertEquals(14, instance.getFactorCount(4));
        assertEquals(0, instance.getFactorCount(5));
        
        // Negative numbers should be handled gracefully.
        assertEquals(0, instance.getFactorCount(-1));
    }

    /**
     * Test of hasFactors method, of class TensorFactorizationMachine.
     */
    @Test
    public void testHasFactors()
    {
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(false, instance.hasFactors(0));
        assertEquals(false, instance.hasFactors(1));
        assertEquals(false, instance.hasFactors(2));
        assertEquals(false, instance.hasFactors(3));
        assertEquals(false, instance.hasFactors(4));
        assertEquals(false, instance.hasFactors(5));
        
        instance.setFactorsPerWay(
            MatrixFactory.getDefault().createMatrix(12, 4),
            null,
            MatrixFactory.getDefault().createMatrix(14, 4));
        assertEquals(false, instance.hasFactors(0));
        assertEquals(false, instance.hasFactors(1));
        assertEquals(true, instance.hasFactors(2));
        assertEquals(false, instance.hasFactors(3));
        assertEquals(true, instance.hasFactors(4));
        assertEquals(false, instance.hasFactors(5));
        
        instance = new TensorFactorizationMachine(40, 7, 0, 14);
        assertEquals(false, instance.hasFactors(0));
        assertEquals(false, instance.hasFactors(1));
        assertEquals(true, instance.hasFactors(2));
        assertEquals(false, instance.hasFactors(3));
        assertEquals(true, instance.hasFactors(4));
        assertEquals(false, instance.hasFactors(5));
        
        // Negative numbers should be handled gracefully.
        assertEquals(false, instance.hasFactors(-1));
    }

    /**
     * Test of getBias method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetBias()
    {
        this.testSetBias();
    }

    /**
     * Test of setBias method, of class TensorFactorizationMachine.
     */
    @Test
    public void testSetBias()
    {
        double bias = 0.0;
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(bias, instance.getBias(), 0.0);
        
        double[] values = {0.4, -0.4, 40, -40};
        for (double value : values)
        {
            bias = value;
            instance.setBias(bias);
            assertEquals(bias, instance.getBias(), 0.0);
        }
    }

    /**
     * Test of getWeights method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetWeights()
    {
        this.testSetWeights();
    }

    /**
     * Test of setWeights method, of class TensorFactorizationMachine.
     */
    @Test
    public void testSetWeights()
    {
        Vector weights = null;
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertSame(weights, instance.getWeights());

        weights = VectorFactory.getSparseDefault().createVector(11);
        instance.setWeights(weights);
        assertSame(weights, instance.getWeights());
        
        weights = null;
        instance.setWeights(weights);
        assertSame(weights, instance.getWeights());
    }

    /**
     * Test of getFactorsPerWay method, of class TensorFactorizationMachine.
     */
    @Test
    public void testGetFactorsPerWay()
    {
        this.testSetFactorsPerWay();
    }

    /**
     * Test of setFactorsPerWay method, of class TensorFactorizationMachine.
     */
    @Test
    public void testSetFactorsPerWay()
    {
        Matrix[] factorsPerWay = null;
        TensorFactorizationMachine instance = new TensorFactorizationMachine();
        assertEquals(0, instance.getFactorsPerWay().length);

        factorsPerWay = new Matrix[]
        {
            MatrixFactory.getSparseDefault().createMatrix(11, 60),
            null,
            MatrixFactory.getSparseDefault().createMatrix(3, 60),
        };
    
        instance.setFactorsPerWay(factorsPerWay);
        assertSame(factorsPerWay, instance.getFactorsPerWay());
        
        // Check that null factors per way is not supported.
        boolean exceptionThrown = false;
        try
        {
            instance.setFactorsPerWay((Matrix[]) null);
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        assertSame(factorsPerWay, instance.getFactorsPerWay());
        
        // Check that the input dimensionality (number of columns) must be equal
        // between factor matrices.
        exceptionThrown = false;
        try
        {
            instance.setFactorsPerWay(
                MatrixFactory.getSparseDefault().createMatrix(11, 60),
                null,
                MatrixFactory.getSparseDefault().createMatrix(3, 61));
        }
        catch (IllegalArgumentException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }
        assertSame(factorsPerWay, instance.getFactorsPerWay());
    }
    
    
    /**
     * Test of computeDiagonalCoefficients method, of class TensorFactorizationMachine.
     */
    @Test
    public void testComputeDiagonalCoefficients()
    {
        double[][] result = computeDiagonalCoefficients(5);
        
        for (int i = 0; i < result.length; i++)
        {
            final double[] coefficients = result[i];
            System.out.print("" + i + ": ");
            for (double value : coefficients)
            {
                System.out.print(" " + value);
            }
            System.out.println();
        }
        fail("The test case is a prototype.");
    }
    
    // TODO: This was old code when the number of expected terms was the number of ways. Not currently used.
    protected static double[][] computeDiagonalCoefficients(
        final int maxWay)
    {
        double[][] coefficients = new double[maxWay + 1][];
        
        for (int i = 0; i <= maxWay; i++)
        {
            coefficients[i] = new double[i + 1];
        }
        
        coefficients[0][0] = 0;
        coefficients[1][0] = 0;
        coefficients[1][1] = 1;
        
        for (int i = 2; i <= maxWay; i++)
        {
            coefficients[i][0] = 0;
            coefficients[i][i] = 1;
            for (int j = 1; j < i; j++)
            {
                coefficients[i][j] = coefficients[i - 1][j - 1] - (i - 1) * coefficients[i - 1][j];
            }
        }
        
        return coefficients;
    }
    
    public double computePairwise(
        final Vector input,
        final Matrix factors,
        final int way)
    {
        final int[] indices = new int[way];
        return computePairwise(input, factors, indices, 0);
    }
    
    public double computePairwise(
        final Vector input,
        final Matrix factors,
        final int[] indices,
        final int index)
    {
        if (index >= indices.length)
        {
            Vector accumulator = VectorFactory.getDenseDefault().createVector(
                factors.getNumRows(), 1.0);

            for (final int i : indices)
            {
                Vector xv = factors.getColumn(i).scale(input.get(i));
                accumulator.dotTimesEquals(xv);
            }
            return accumulator.sum();
        }
        else
        {
            double result = 0.0;
            final int d = input.getDimensionality();
            final int start = index == 0 ? 0 : indices[index - 1] + 1;
            for (int i = start; i < d; i++)
            {
                indices[index] = i;
                result += computePairwise(input, factors, indices, index + 1);

            }
            return result;
        }
    }
    
    public Matrix computePairwiseGradients(
        final Vector input,
        final Matrix factors,
        final int way)
    {
        final int d = input.getDimensionality();
        final int k = factors.getNumRows();
        Matrix gradients = MatrixFactory.getDenseDefault().createMatrix(k, d);
        for (int i = 0; i < d; i++)
        {
            gradients.setColumn(i, computePairwiseGradient(
                input, factors, way, i));
        }
        return gradients;
    }
    
    public Vector computePairwiseGradient(
        final Vector input,
        final Matrix factors,
        final int way,
        final int featureIndex)
    {
        final int[] indices = new int[way];
        return computePairwiseGradient(input, factors, featureIndex, indices, 0);
    }
    
    public Vector computePairwiseGradient(
        final Vector input,
        final Matrix factors,
        final int featureIndex,
        final int[] indices,
        final int index)
    {
        if (index >= indices.length)
        {
            Vector accumulator = VectorFactory.getDenseDefault().createVector(
                factors.getNumRows(), 1.0);
            double inputValueForIndex = 0.0;
            for (final int i : indices)
            {
                if (i == featureIndex)
                {
                    inputValueForIndex = input.get(i);
                }
                else
                {
                    Vector xv = factors.getColumn(i).scale(input.get(i));
                    accumulator.dotTimesEquals(xv);
                }
            }
            return accumulator.scale(inputValueForIndex);
        }
        else
        {
            Vector result = VectorFactory.getDenseDefault().createVector(
                factors.getNumRows());
            final int d = input.getDimensionality();
            final int start = index == 0 ? 0 : indices[index - 1] + 1;
            for (int i = start; i < d; i++)
            {
                indices[index] = i;
                final Vector part = computePairwiseGradient(input, factors, 
                    featureIndex, indices, index + 1);
                result.plusEquals(part);
            }
            return result;
        }
    }
    
// TODO: Make this a utility function somewhere.
    public static int[][] createPartitions(
        final int n)
    {
        ArgumentChecker.assertIsPositive("n", n);
// TEMP: Code from web... refactor.        
        final LinkedList<int[]> partitions = new LinkedList<>();
        final int[] a = new int[n + 1];
        a[1] = n;
        
        int k = 1;
        while (k != 0)
        {
            final int x = a[k - 1] + 1;
            int y = a[k] - 1;
            k -= 1;
            
            while (x <= y)
            {
                a[k] = x;
                y -= x;
                k += 1;
            }
            a[k] = x + y;
            
            final int[] part = Arrays.copyOfRange(a, 0, k + 1);
            partitions.add(part);
        }
        
        return partitions.toArray(new int[partitions.size()][]);
    }
    
    @Test
    public void testCreatePartitions()
    {
        for (int i = 1; i <= 10; i++)
        {
            System.out.println("Partitions of " + i);
            int[][] result = createPartitions(i);
            System.out.println("There are " + result.length + " partitions of " + i);
            for (int[] part : result)
            {
                System.out.print("{");
                boolean first = true;
                for (int x : part)
                {
                    if (!first)
                    {
                        System.out.print(", ");
                    }
                    else
                    {
                        first = false;
                    }
                    System.out.print(x);
                }
                System.out.print("}, ");
            }
            System.out.println();
        }
        
        fail("Not yet implemented");
    }
    
    @Test
    public void testStatics()
    {
        for (int way = 1; way < TensorFactorizationMachine.COEFFICIENTS_PER_WAY.length; way++)
        {
            System.out.println("Way: " + way);
            
            int terms = TensorFactorizationMachine.COEFFICIENTS_PER_WAY[way].length;
            for (int i = 0; i < terms; i++)
            {
                System.out.println(TensorFactorizationMachine.COEFFICIENTS_PER_WAY[way][i] + "\t"
                    + Arrays.toString(TensorFactorizationMachine.PARTITIONS_PER_WAY[way][i]));
            }
        }
    }
    
}
