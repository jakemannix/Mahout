package org.apache.mahout.math;

/**
 * 
 * Abstract base class of LinearOperator implementations. Provides default implementations of some of
 * the interface methods. Many of these default implementations are rather naive and inefficient and
 * should be overridden by specific subclasses.
 *
 */
public abstract class AbstractLinearOperator implements LinearOperator {
	
  /**
   * 
   * @return The dimension of the operator's range. 
   */
  
  public final int rangeDimension() {
	  return numRows();
  }
	
  /**
   * Semantically the same as rangeDimension(), but with a more familiar name for matrices.
   * 
   * @return The number of rows in the matrix representation of this operator.
   */
  
  public abstract int numRows();
	
  /**
   * 
   * @return The dimension of the operator's domain.
   */

  public final int domainDimension() {
    return numCols();
  }
  
  /**
   * Semantically the same as domainDimension(), but with a more familiar name for matrices.
   * 
   * @return The number of columns in the matrix representation of this operator.
   */
  
  public abstract int numCols();
	
  /**
   * 
   * Applies the operator to a vector of cardinality equal to the domain dimension and returns a new vector 
   * of dimension equal to range dimension that is the result of applying the transformation to the input.
   *
   * @param v a vector with cardinality equal to getNumCols() of the recipient
   * @return a new vector (typically a DenseVector)
   * @throws CardinalityException if this.domainDimension() != v.size()
   */
  
  public abstract Vector times(Vector v);

  public Vector timesSquared(Vector v) {
    return times(times(v));
  }

  /**
   * Returns a new linear operator which is the composition of this operator and another. I.e. if this object
   * represents linear operator A(x), then A.times(B) returns the operator A(B(x)). For matrices, this operation
   * is equivalent to matrix multiplication. 
   * 
   * This default implementation of this method may result in a linear operator with a fairly inefficient
   * times(Vector) method, since it works by naively applying the other operator to any input, then applying
   * this operator to the result. 
   * 
   * @param other The operator with which to compose this operator.
   * @return The composed operator.
   * @throws CardinalityException if the domain dimension of this operator is not equal to the range dimension of
   * the other.
   */
    
  public LinearOperator times(LinearOperator other) {
    return new ProductLinearOperator(this, other);
  }
  
  /**
   * Returns a new linear operator which is the sum of this operator and another. The resulting operator is the 
   * the equivalent of applying both operators to a given input, then summing the result. For matrices, this
   * operation is equivalent to matrix addition.
   * 
   * This default implementation of this method may result in a linear operator with a fairly inefficient
   * times(Vector) method, since it works by naively both operators separately to any input, then summing
   * the result. 
   * 
   * @param other The operator with which to add this operator.
   * @return The summed operator.
   * @throws CardinalityException if the two operators do not have same domain and range dimensions.
   */

  public LinearOperator plus(LinearOperator other) {
    return new SumLinearOperator(this, other);
  }
  
  /**
   * Returns a new linear operator which is the result of scaling this operator by a constant. The resulting operator
   * is the equivalent of applying this operator to a vector, then multiplying the resulting vector by the given scalar.
   * For matrices, this operation is scalar multiplication.
   *
   * @param scalar The scaling constant.
   * @return The scaled operator.
   */
  
  public LinearOperator scale(double scalar) {
    return new ScaledLinearOperator(this, scalar);
  }
}
