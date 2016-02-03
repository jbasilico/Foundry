/*
 * File:                DatasetUtilTest.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 *
 * Copyright September 19, 2007, Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this work by
 * or on behalf of the U.S. Government. Export of this program may require a
 * license from the United States Government. See CopyrightHistory.txt for
 * complete details.
 *
 */

package gov.sandia.cognition.learning.data;

import gov.sandia.cognition.math.matrix.DimensionalityMismatchException;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.Vector;
import java.util.ArrayList;
import gov.sandia.cognition.util.DefaultPair;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Random;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.mtj.Vector2;
import gov.sandia.cognition.math.matrix.mtj.Vector3;
import gov.sandia.cognition.statistics.DataDistribution;
import java.util.List;
import java.util.Map;
import java.util.Set;
import junit.framework.TestCase;

/**
 *
 * @author Kevin R. Dixon
 */
public class DatasetUtilTest
    extends TestCase
{

    /** The random number generator for the tests. */
    protected Random random = new Random(1);

    /**
     * 
     * @param testName
     */
    public DatasetUtilTest(
        String testName )
    {
        super( testName );
    }

    /**
     * Test of appendBias method, of class gov.sandia.cognition.learning.util.data.DatasetUtil.
     */
    public void testAppendBias()
    {
        System.out.println( "appendBias" );

        int M = (int) (random.nextDouble() * 5) + 2;
        int num = (int) (random.nextDouble() * 10000) + 1000;

        ArrayList<Vector> dataset = new ArrayList<Vector>( num );
        for (int n = 0; n < num; n++)
        {
            dataset.add( VectorFactory.getDefault().createUniformRandom( M, -1, 1, random ) );
        }

        ArrayList<Vector> result = DatasetUtil.appendBias( dataset );
        assertEquals( dataset.size(), result.size() );
        for (int n = 0; n < num; n++)
        {
            assertEquals( M + 1, result.get( n ).getDimensionality() );
            assertEquals( dataset.get( n ), result.get( n ).subVector( 0, M - 1 ) );
            assertEquals( 1.0, result.get( n ).getElement( M ) );
        }

    }

    /**
     * Test of decoupleVectorPairDataset method, of class gov.sandia.cognition.learning.util.data.DatasetUtil.
     */
    public void testDecoupleVectorPairDataset()
    {
        System.out.println( "decoupleVectorPairDataset" );

        int M = (int) (random.nextDouble() * 5) + 2;
        int num = (int) (random.nextDouble() * 10000) + 1000;

        ArrayList<WeightedInputOutputPair<Vector, Vector>> dataset =
            new ArrayList<WeightedInputOutputPair<Vector, Vector>>( num );
        for (int n = 0; n < num; n++)
        {
            Vector input = VectorFactory.getDefault().createUniformRandom( M, -1, 1, random );
            Vector output = VectorFactory.getDefault().createUniformRandom( M, -1, 1, random );
            dataset.add( new DefaultWeightedInputOutputPair<Vector, Vector>(
                input, output, random.nextDouble() ) );
        }

        ArrayList<ArrayList<InputOutputPair<Double, Double>>> result =
            DatasetUtil.decoupleVectorPairDataset( dataset );
        assertEquals( M, result.size() );
        for (int i = 0; i < M; i++)
        {
            assertEquals( dataset.size(), result.get( i ).size() );
            for (int n = 0; n < num; n++)
            {
                assertEquals( dataset.get( n ).getInput().getElement( i ), result.get( i ).get( n ).getInput() );
                assertEquals( dataset.get( n ).getOutput().getElement( i ), result.get( i ).get( n ).getOutput() );
            }
        }


    }

    /**
     * Test of decoupleVectorDataset method, of class gov.sandia.cognition.learning.util.data.DatasetUtil.
     */
    public void testDecoupleVectorDataset()
    {
        System.out.println( "decoupleVectorDataset" );

        int M = (int) (random.nextDouble() * 5) + 2;
        int num = (int) (random.nextDouble() * 10000) + 1000;

        ArrayList<Vector> dataset = new ArrayList<Vector>( num );
        for (int n = 0; n < num; n++)
        {
            dataset.add( VectorFactory.getDefault().createUniformRandom( M, -1, 1, random ) );
        }

        ArrayList<ArrayList<Double>> result =
            DatasetUtil.decoupleVectorDataset( dataset );
        assertEquals( M, result.size() );
        for (int i = 0; i < M; i++)
        {
            assertEquals( dataset.size(), result.get( i ).size() );
            for (int n = 0; n < num; n++)
            {
                assertEquals( dataset.get( n ).getElement( i ), result.get( i ).get( n ) );
            }
        }
    }

    /**
     * Test of splitDatasets method, of class gov.sandia.cognition.learning.util.data.DatasetUtil.
     */
    public void testSplitDatasets()
    {
        System.out.println( "splitDatasets" );

        Collection<InputOutputPair<Double, Boolean>> data =
            new LinkedList<InputOutputPair<Double, Boolean>>();
        LinkedList<Double> nt = new LinkedList<Double>();
        LinkedList<Double> nf = new LinkedList<Double>();
        int num = random.nextInt( 1000 ) + 100;

        for (int i = 0; i < num; i++)
        {
            boolean c = random.nextBoolean();
            double v = random.nextDouble();
            if (c == true)
            {
                nt.add( v );
            }
            else
            {
                nf.add( v );
            }
            data.add( new DefaultInputOutputPair<Double, Boolean>( v, c ) );
        }

        DefaultPair<LinkedList<Double>, LinkedList<Double>> result = DatasetUtil.splitDatasets( data );
        assertEquals( nt.size(), result.getFirst().size() );
        assertEquals( nf.size(), result.getSecond().size() );
        for (int i = 0; i < nt.size(); i++)
        {
            assertEquals( nt.get( i ), result.getFirst().get( i ) );
        }
        for (int i = 0; i < nf.size(); i++)
        {
            assertEquals( nf.get( i ), result.getSecond().get( i ) );
        }
    }

    /**
     * Test of splitOnOutput method, of class DatasetUtil.
     */
    public void testSplitOnOutput()
    {
        Collection<InputOutputPair<Double, Boolean>> data =
            new LinkedList<InputOutputPair<Double, Boolean>>();
        LinkedList<Double> nt = new LinkedList<Double>();
        LinkedList<Double> nf = new LinkedList<Double>();
        int num = random.nextInt( 1000 ) + 100;

        for (int i = 0; i < num; i++)
        {
            boolean c = random.nextBoolean();
            double v = random.nextDouble();
            if (c == true)
            {
                nt.add( v );
            }
            else
            {
                nf.add( v );
            }
            data.add( new DefaultInputOutputPair<Double, Boolean>( v, c ) );
        }

        Map<Boolean, List<Double>> map = DatasetUtil.splitOnOutput(data);
        assertEquals(nt.size(), map.get(true).size());
        assertEquals(nf.size(), map.get(false).size());
        for (int i = 0; i < nt.size(); i++)
        {
            assertEquals( nt.get( i ), map.get(true).get( i ) );
        }
        for (int i = 0; i < nf.size(); i++)
        {
            assertEquals( nf.get( i ), map.get(false).get( i ) );
        }
    }

    /**
     * Test of computeOuterProductDataMatrix method, of class gov.sandia.cognition.learning.util.data.DatasetUtil.
     */
    public void testComputeOuterProductDataMatrix()
    {
        System.out.println( "computeOuterProductDataMatrix" );

        int num = 10;
        int M = 3;
        double r = 1;
        Matrix X = MatrixFactory.getDefault().createMatrix( M, num );
        ArrayList<Vector> data = new ArrayList<Vector>( num );
        for (int i = 0; i < num; i++)
        {
            Vector x = VectorFactory.getDefault().createUniformRandom( M, -r, r, random );
            data.add( x );
            X.setColumn( i, x );
        }

        Matrix XXt = X.times( X.transpose() );

        Matrix XXthat = DatasetUtil.computeOuterProductDataMatrix( data );
        if (XXt.equals( XXthat, 1e-5 ) == false)
        {
            assertEquals( XXt, XXthat );
        }
    }

    /**
     * Test of computeOutputMean method, of class DatasetUtil.
     */
    public void testComputeOutputMean()
    {
        Collection<InputOutputPair<Object, Double>> data = null;
        assertEquals( 0.0, DatasetUtil.computeOutputMean( data ) );

        data = new LinkedList<InputOutputPair<Object, Double>>();
        assertEquals( 0.0, DatasetUtil.computeOutputMean( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 4.0 ) );
        assertEquals( 4.0, DatasetUtil.computeOutputMean( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 4.0 ) );
        assertEquals( 4.0, DatasetUtil.computeOutputMean( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 2.0 ) );
        data.add( new DefaultInputOutputPair<Object, Double>( null, 2.0 ) );
        assertEquals( 3.0, DatasetUtil.computeOutputMean( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 0.0 ) );
        assertEquals( 2.4, DatasetUtil.computeOutputMean( data ), 0.001 );


        data.add( new DefaultInputOutputPair<Object, Double>( null, -10.0 ) );
        assertEquals( 0.333, DatasetUtil.computeOutputMean( data ), 0.001 );
    }

    /**
     * Test of computeVariance method, of class DatasetUtil.
     */
    public void testComputeOutputVariance()
    {
        Collection<InputOutputPair<Object, Double>> data = null;
        assertEquals( 0.0, DatasetUtil.computeOutputVariance( data ) );

        data = new LinkedList<InputOutputPair<Object, Double>>();
        assertEquals( 0.0, DatasetUtil.computeOutputVariance( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 4.0 ) );
        assertEquals( 0.0, DatasetUtil.computeOutputVariance( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 4.0 ) );
        assertEquals( 0.0, DatasetUtil.computeOutputVariance( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 2.0 ) );
        data.add( new DefaultInputOutputPair<Object, Double>( null, 2.0 ) );
        assertEquals( 1.0, DatasetUtil.computeOutputVariance( data ) );

        data.add( new DefaultInputOutputPair<Object, Double>( null, 0.0 ) );
        assertEquals( 2.24, DatasetUtil.computeOutputVariance( data ), 0.001 );


        data.add( new DefaultInputOutputPair<Object, Double>( null, -10.0 ) );
        assertEquals( 23.222, DatasetUtil.computeOutputVariance( data ), 0.001 );
    }

    /**
     * Test of findUniqueOutputs method, of class DatasetUtil.
     */
    public void testFindUniqueOutputs()
    {
        Collection<InputOutputPair<Object, String>> data = null;
        Set<String> result = DatasetUtil.findUniqueOutputs(data);
        assertTrue(result.isEmpty());

        data = new LinkedList<InputOutputPair<Object, String>>();
        result = DatasetUtil.findUniqueOutputs(data);
        assertTrue(result.isEmpty());

        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.findUniqueOutputs(data);
        assertEquals(1, result.size());
        assertTrue(result.contains("one"));

        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.findUniqueOutputs(data);
        assertEquals(1, result.size());
        assertTrue(result.contains("one"));

        data.add(new DefaultInputOutputPair<Object, String>(null, "two"));
        result = DatasetUtil.findUniqueOutputs(data);
        assertEquals(2, result.size());
        assertEquals("one", result.toArray()[0]);
        assertEquals("two", result.toArray()[1]);
        

        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        result = DatasetUtil.findUniqueOutputs(data);
        assertEquals(3, result.size());
        assertEquals("one", result.toArray()[0]);
        assertEquals("two", result.toArray()[1]);
        assertEquals("another", result.toArray()[2]);
        
        data.add(new DefaultInputOutputPair<Object, String>(null, "two"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.findUniqueOutputs(data);
        assertEquals(3, result.size());
        assertEquals("one", result.toArray()[0]);
        assertEquals("two", result.toArray()[1]);
        assertEquals("another", result.toArray()[2]);
    }


    /**
     * Test of countOutputValues method, of class DatasetUtil.
     */
    public void testCountOutputValues()
    {
        Collection<InputOutputPair<Object, String>> data = null;
        DataDistribution<String> result = DatasetUtil.countOutputValues(data);
        assertTrue(result.isEmpty());

        data = new LinkedList<InputOutputPair<Object, String>>();
        result = DatasetUtil.countOutputValues(data);
        assertTrue(result.isEmpty());

        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.countOutputValues(data);
        assertEquals(1, result.getDomainSize());
        assertEquals(1.0, result.getTotal());
        assertEquals(1.0, result.get("one"));

        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.countOutputValues(data);
        assertEquals(1, result.getDomainSize());
        assertEquals(2.0, result.getTotal());
        assertEquals(2.0, result.get("one"));

        data.add(new DefaultInputOutputPair<Object, String>(null, "two"));
        result = DatasetUtil.countOutputValues(data);
        assertEquals(2, result.getDomainSize());
        assertEquals(3.0, result.getTotal());
        assertEquals(2.0, result.get("one"));
        assertEquals(1.0, result.get("two"));


        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        result = DatasetUtil.countOutputValues(data);
        assertEquals(3, result.getDomainSize());
        assertEquals(6.0, result.getTotal());
        assertEquals(2.0, result.get("one"));
        assertEquals(1.0, result.get("two"));
        assertEquals(3.0, result.get("another"));

        data.add(new DefaultInputOutputPair<Object, String>(null, "two"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "another"));
        data.add(new DefaultInputOutputPair<Object, String>(null, "one"));
        result = DatasetUtil.countOutputValues(data);
        assertEquals(3, result.getDomainSize());
        assertEquals(9.0, result.getTotal());
        assertEquals(3.0, result.get("one"));
        assertEquals(2.0, result.get("two"));
        assertEquals(4.0, result.get("another"));
    }
    
    /**
     * Test of findMinOutput method, of class DatasetUtil.
     */
    public void testFindMinOutput()
    {
        Collection<InputOutputPair<Object, Double>> data = null;
        assertEquals(Double.POSITIVE_INFINITY, DatasetUtil.findMinOutput(data));

        data = new LinkedList<>();
        assertEquals(Double.POSITIVE_INFINITY, DatasetUtil.findMinOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 2.0));
        assertEquals(2.0, DatasetUtil.findMinOutput(data));
        
        data.add(new DefaultInputOutputPair<>(null, 4.0));
        assertEquals(2.0, DatasetUtil.findMinOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 4.0));
        assertEquals(2.0, DatasetUtil.findMinOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 2.0));
        assertEquals(2.0, DatasetUtil.findMinOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 0.0));
        assertEquals(0.0, DatasetUtil.findMinOutput(data), 0.001);

        data.add(new DefaultInputOutputPair<>(null, -10.0));
        assertEquals(-10.0, DatasetUtil.findMinOutput(data), 0.001);
        
        data.add(new DefaultInputOutputPair<>(null, 11.0));
        assertEquals(-10.0, DatasetUtil.findMinOutput(data), 0.001);
    }
    
    /**
     * Test of findMaxOutput method, of class DatasetUtil.
     */
    public void testFindMaxOutput()
    {
        Collection<InputOutputPair<Object, Double>> data = null;
        assertEquals(Double.NEGATIVE_INFINITY, DatasetUtil.findMaxOutput(data));

        data = new LinkedList<>();
        assertEquals(Double.NEGATIVE_INFINITY, DatasetUtil.findMaxOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 2.0));
        assertEquals(2.0, DatasetUtil.findMaxOutput(data));
        
        data.add(new DefaultInputOutputPair<>(null, 4.0));
        assertEquals(4.0, DatasetUtil.findMaxOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 4.0));
        assertEquals(4.0, DatasetUtil.findMaxOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 2.0));
        assertEquals(4.0, DatasetUtil.findMaxOutput(data));

        data.add(new DefaultInputOutputPair<>(null, 0.0));
        assertEquals(4.0, DatasetUtil.findMaxOutput(data), 0.001);

        data.add(new DefaultInputOutputPair<>(null, -10.0));
        assertEquals(4.0, DatasetUtil.findMaxOutput(data), 0.001);
        
        data.add(new DefaultInputOutputPair<>(null, 11.0));
        assertEquals(11.0, DatasetUtil.findMaxOutput(data), 0.001);
    }

    /**
     * Test of inputsList method, of class DatasetUtil.
     */
    public void testInputsList()
    {
        String a = "something";
        String b = "b";
        String c = "";

        ArrayList<InputOutputPair<String, String>> data = new ArrayList<InputOutputPair<String, String>>();
        data.add(new DefaultInputOutputPair<String, String>(a, "output"));
        data.add(new DefaultInputOutputPair<String, String>(b, "output"));
        data.add(new DefaultInputOutputPair<String, String>(c, "output"));

        List<String> inputs = DatasetUtil.inputsList(data);
        assertEquals(3, inputs.size());
        assertSame(a, inputs.get(0));
        assertSame(b, inputs.get(1));
        assertSame(c, inputs.get(2));
    }

    /**
     * Test of outputsList method, of class DatasetUtil.
     */
    public void testOutputsList()
    {
        String a = "something";
        String b = "b";
        String c = "";

        ArrayList<InputOutputPair<String, String>> data = new ArrayList<InputOutputPair<String, String>>();
        data.add(new DefaultInputOutputPair<String, String>("input", a));
        data.add(new DefaultInputOutputPair<String, String>("input", b));
        data.add(new DefaultInputOutputPair<String, String>("input", c));

        List<String> outputs = DatasetUtil.outputsList(data);
        assertEquals(3, outputs.size());
        assertSame(a, outputs.get(0));
        assertSame(b, outputs.get(1));
        assertSame(c, outputs.get(2));
    }

    /**
     * Test of getInputDimensionality method of class DatasetUtil.
     */
    public void testGetInputDimensionality()
    {
        ArrayList<InputOutputPair<Vector, Object>> data =
            new ArrayList<InputOutputPair<Vector, Object>>();
        assertEquals(-1, DatasetUtil.getInputDimensionality(data));

        data.add(null);
        assertEquals(-1, DatasetUtil.getInputDimensionality(data));

        data.add(new DefaultInputOutputPair<Vector, Object>(null, null));
        assertEquals(-1, DatasetUtil.getInputDimensionality(data));

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector3(), null));
        assertEquals(3, DatasetUtil.getInputDimensionality(data));

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector2(), null));
        assertEquals(3, DatasetUtil.getInputDimensionality(data));
    }

    /**
     * Test of assertInputDimensionalitiesAllEqual method of class DatasetUtil.
     */
    public void testAssertInputDimensionalitiesAreEqual()
    {
        LinkedList<InputOutputPair<Vector, Object>> data =
            new LinkedList<InputOutputPair<Vector, Object>>();

        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(null);
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(new DefaultInputOutputPair<Vector, Object>(null, null));
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector3(), null));
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector3(), null));
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector2(), null));

        boolean exceptionThrown = false;
        try
        {
            DatasetUtil.assertInputDimensionalitiesAllEqual(data);
        }
        catch (DimensionalityMismatchException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }

        data.clear();
        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector2(), null));
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);

        data.add(new DefaultInputOutputPair<Vector, Object>(new Vector2(), null));
        DatasetUtil.assertInputDimensionalitiesAllEqual(data);
        DatasetUtil.assertInputDimensionalitiesAllEqual(data, 2);

        exceptionThrown = false;
        try
        {
            DatasetUtil.assertInputDimensionalitiesAllEqual(data, 3);
        }
        catch (DimensionalityMismatchException e)
        {
            exceptionThrown = true;
        }
        finally
        {
            assertTrue(exceptionThrown);
        }

    }


    /**
     * Test of sumWeights method, of class DatasetUtil.
     */
    public void testSumWeights()
    {
        ArrayList<InputOutputPair<String, String>> data = new ArrayList<InputOutputPair<String, String>>();

        assertEquals(0.0, DatasetUtil.sumWeights(data), 0.0);

        data.add(new DefaultInputOutputPair<String, String>("input1", "a"));
        assertEquals(0.0, DatasetUtil.sumWeights(data), 1.0);

        data.add(new DefaultInputOutputPair<String, String>("input2", "b"));
        assertEquals(0.0, DatasetUtil.sumWeights(data), 2.0);

        data.add(new DefaultWeightedInputOutputPair<String, String>("input3", "c", 0.7));

        assertEquals(0.0, DatasetUtil.sumWeights(data), 2.7);
    }
}
