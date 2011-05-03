package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

public class SumLinearOperator extends AbstractLinearOperator {
  private LinearOperator a;
  private LinearOperator b;
  
  public SumLinearOperator(LinearOperator a, LinearOperator b) {
    if (a.domainDimension() != b.domainDimension()) {
      throw new CardinalityException(a.domainDimension(), b.domainDimension());
    }
    if (a.rangeDimension() != b.rangeDimension()) {
      throw new CardinalityException(a.rangeDimension(), b.rangeDimension());
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
    return a.numCols();
  }
  
  @Override
  public Vector times(Vector v) {
    Vector result1 = a.times(v);
    Vector result2 = b.times(v);
    result1.assign(result2, Functions.PLUS);
    return result1;
  }
}
