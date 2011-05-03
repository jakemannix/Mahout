package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

/**
 * Linear operator that implements the result of scaling another linear operator by a constant.
 *
 */

public class ScaledLinearOperator extends AbstractLinearOperator {
  private LinearOperator a;
  private double scalar;
  
  public ScaledLinearOperator(LinearOperator a, double scalar) {
    this.a = a;
    this.scalar = scalar;
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
    Vector result = a.times(v);
    result.assign(Functions.MULT, scalar);
    return result;
  }  
}
