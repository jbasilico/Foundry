/*
 * File:                VectorUnionIterator.java
 * Authors:             Kevin R. Dixon
 * Company:             Sandia National Laboratories
 * Project:             Cognitive Foundry
 *
 * Copyright March 20, 2006, Sandia Corporation.  Under the terms of Contract
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this work by
 * or on behalf of the U.S. Government. Export of this program may require a
 * license from the United States Government. See CopyrightHistory.txt for
 * complete details.
 *
 */

package gov.sandia.cognition.math.matrix;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Iterator that returns all non-zero entries for either underlying Vector. This
 * iterator is implemented in a way such that it reuses the same entry between
 * calls to next() so it should be copied.
 *
 * @author  Kevin R. Dixon
 * @author  Justin Basilico
 * @since   1.0
 */
public class VectorUnionIterator
    extends Object
    implements Iterator<TwoVectorEntry>
{
    
    /** The iterator for the first vector. */
    protected final Iterator<VectorEntry> firstIterator;
    
    /** The iterator for the second vector. */
    protected final Iterator<VectorEntry> secondIterator;
    
    /** The internal entry data structure. */
    protected TwoVectorEntry internalEntry;
    
    /** The current entry for the first iterator. */
    protected VectorEntry firstEntry;
    
    /** The current entry for the second iterator. */
    protected VectorEntry secondEntry;
    
    /**
     * Creates a new instance of VectorUnionIterator.
     *
     * @param  first The first Vector.
     * @param  second The second Vector.
     */
    public VectorUnionIterator(
        final Vector first,
        final Vector second)
    {
        super();
        
        this.firstIterator = first.iterator();
        this.secondIterator = second.iterator();
        this.internalEntry = new DefaultTwoVectorEntry(first, second, -1);
        this.firstEntry = VectorUtil.nextNonZeroOrNull(this.firstIterator);
        this.secondEntry = VectorUtil.nextNonZeroOrNull(this.secondIterator);
    }
    
    @Override
    public boolean hasNext()
    {
        // The entries are already at the next position, so as long as there
        // is one, there is more to iterate over.
        return this.firstEntry != null || this.secondEntry != null;
    }

    @Override
    public TwoVectorEntry next()
    {
        // Figure out the index of the two vectors to use next.
        final boolean hasFirst = this.firstEntry != null;
        final boolean hasSecond = this.secondEntry != null;
        int index = -1;
        if (hasFirst && hasSecond)
        {
            // Normal case where both have entries. The next value is the
            // minimum index.
            final int firstIndex = this.firstEntry.getIndex();
            final int secondIndex = this.secondEntry.getIndex();
            
            if (firstIndex <= secondIndex)
            {
                // Advance the first iterator since it is at the minimum index.
                index = firstIndex;
                this.firstEntry = VectorUtil.nextNonZeroOrNull(this.firstIterator);
            }
            
            if (secondIndex <= firstIndex)
            {
                // Advance the second iterator since it is a the minimum index.
                index = secondIndex;
                this.secondEntry = VectorUtil.nextNonZeroOrNull(this.secondIterator);
            }
            
            // Note that above it may advance both iterators if they're at the
            // same index. Technically index gets set twice also, but that 
            // doesn't really matter since they're the same.
        }
        else if (hasFirst)
        {
            // This means the second iterator ran out. Just keep going on the
            // first one.
            index = this.firstEntry.getIndex();
            this.firstEntry = VectorUtil.nextNonZeroOrNull(this.firstIterator);
        }
        else if (hasSecond)
        {
            // This means the first iterator ran out. Just keep going on the 
            // second one.
            index = this.secondEntry.getIndex();
            this.secondEntry = VectorUtil.nextNonZeroOrNull(this.secondIterator);
        }
        else
        {
            // Both iterators ran out. No more data, so an exception.
            // Index is set to -1 here to indicate the entry is now
            // bad.
            this.internalEntry.setIndex(-1);
            throw new NoSuchElementException();
        }
        
        // Update the internal entry to the new index.
        this.internalEntry.setIndex(index);
        return this.internalEntry;
    }

    @Override
    public void remove()
    {
        if (this.internalEntry.getIndex() < 0)
        {
            // Indicates there is no more internal entries.
            throw new NoSuchElementException();
        }
        
        this.internalEntry.setFirstValue(0.0);
        this.internalEntry.setSecondValue(0.0);
    }
    
}
