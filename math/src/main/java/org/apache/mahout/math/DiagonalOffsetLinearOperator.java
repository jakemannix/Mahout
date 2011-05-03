package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

/**
 * 
 * Implements a linear operator that is the sum of a given operator and a scaling operator (one that merely
 * multiplies each element of a vector by a fixed but not-necessarily equal amount). For matrices, this is
 * equivalent to the matrix (A + D), where A is a general matrix and D is a diagonal matrix.
 * 
 * A common special case of this operator combines a general linear operator with a uniform scaling operator: 
 * for matrices, A + cI, a square matrix plus a scaled identity matrix of the same size.
 * 
 * This operator can only be created from linear operators that have domains and ranges of the same size.
 *
 */

public class DiagonalOffsetLinearOperator extends AbstractLinearOperator {
  private final LinearOperator matrix;
  private final Vector diagonalOffset;

  /**
   * 
   * Creates a diagonal offset matrix by adding a scaled identity matrix to it.
   * 
   * @param matrix The source matrix.
   * @param identityScaleOffset The scalar value used to scale the identity matrix added to the source.
   * @throws IllegalArgmentException if the provided matrix is not square. 
   */
  public DiagonalOffsetLinearOperator(LinearOperator matrix, double identityScaleOffset) {
    if (matrix.numRows() != matrix.numCols()) {
      throw new IllegalArgumentException("Only square matrices can be used to make a diagonal offset matrix.");
    }
    this.matrix = matrix;
    this.diagonalOffset = new DenseVector(matrix.numCols());
    for (int i = 0; i < matrix.numCols(); ++i) {
      this.diagonalOffset.setQuick(i, identityScaleOffset);
    }
  }

  /**
   * Creates a diagonal offset matrix by adding a diagonal matrix specified as a vector of the diagonal elements.
   * 
   * @param matrix The source matrix for the tranformation.
   * @param diagonalOffset A vector containing the diagonal elements of the offset matrix to add.
   * @throws IllegalArgumentException if the provided source matrix is not square.
   * @throws CardinalityException if the diagonal offset vector size is not equal to the source matrix number of rows/cols.
   */
  public DiagonalOffsetLinearOperator(LinearOperator matrix, Vector diagonalOffset) {
    if (matrix.numRows() != matrix.numCols()) {
      throw new IllegalArgumentException("Only square matrices can be used to make a diagonal offset matrix.");
    }
    if (diagonalOffset.size() != matrix.numCols()) {
      throw new CardinalityException(matrix.numCols(), diagonalOffset.size());
    }
    this.matrix = matrix;
    this.diagonalOffset = diagonalOffset;
  }
  
  @Override
	public int numRows() {
    return matrix.numRows();
	}

	@Override
	public int numCols() {
	  return matrix.numCols();
	}
	
	public LinearOperator getMatrix() {
	  return matrix;
	}

	public Vector getDiagonalOffset() {
	  return diagonalOffset;
	}
	
	@Override
	public Vector times(Vector v) {
	  Vector product = matrix.times(v);
	  product.assign(diagonalOffset, Functions.PLUS);
	  return product;
	}
}
