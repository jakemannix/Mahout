package org.apache.mahout.math.decomposer.lanczos;

import java.io.File;

import org.apache.mahout.math.LinearOperator;
import org.apache.mahout.math.Vector;

public class DiskBackedLanczosState extends LanczosState {

  protected final File dir;

  public DiskBackedLanczosState(LinearOperator corpus, int numCols, boolean isSymmetric, int desiredRank,
      Vector initialVector, File dir) {
    super(corpus, numCols, isSymmetric, desiredRank, initialVector);
    this.dir = dir;
  }

  @Override
  protected void intitializeBasisAndSingularVectors(int rank, int numCols) {
    // TODO
  }

  @Override
  public Vector getBasisVector(int i) {
    return super.getBasisVector(i);
  }

  @Override
  public Vector getRightSingularVector(int i) {
    return super.getRightSingularVector(i);
  }

  @Override
  public void setBasisVector(int i, Vector vector) {
    super.setBasisVector(i, vector);
  }

  @Override
  public void setRightSingularVector(int i, Vector vector) {
    super.setRightSingularVector(i, vector);
  }

}