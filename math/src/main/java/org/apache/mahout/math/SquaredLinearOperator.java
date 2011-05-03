package org.apache.mahout.math;

/**
 * 
 * Linear operator that takes another operator A and yields the operator A'A.
 *
 */

public class SquaredLinearOperator extends AbstractLinearOperator {
  private final LinearOperator operator;
  
  public SquaredLinearOperator(LinearOperator operator) {
    this.operator = operator;
  }
  
  @Override
  public int numRows() {
    return operator.numCols();
  }
  
  @Override
  public int numCols() {
    return operator.numCols();
  }
  
  @Override
  public Vector times(Vector v) {
    return operator.timesSquared(v);
  }  
}
