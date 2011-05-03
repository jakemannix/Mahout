package org.apache.mahout.math;

/**
 * Linear operator that implements the product (composition) of two linear operators. This implementation of
 * the composition operation is general but very naive: it simply applies the second operator to a vector, 
 * then applies the first operator to the result. Specific subclasses of LinearOperator may want to use
 * a more direct implementation of composition/multiplication.
 * 
 */

public class ProductLinearOperator extends AbstractLinearOperator {
  private LinearOperator a;
  private LinearOperator b;
  
  public ProductLinearOperator(LinearOperator a, LinearOperator b) {
    if (a.domainDimension() != b.rangeDimension()) {
      throw new CardinalityException(a.domainDimension(), b.rangeDimension()); 
    }

    this.a = a;
    this.b = b;
  }
  
  @Override
  public int numRows() {
    return a.numRows();
  }
  
  @Override
  public int numCols() {
    return b.numCols();
  }
  
  @Override
  public Vector times(Vector v) {
    return a.times(b.times(v));
  }
}
