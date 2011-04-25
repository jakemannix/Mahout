package org.apache.mahout.math.decomposer.lanczos;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

import java.io.File;

public class DiskBackedLanczosState extends LanczosState {

  protected final File dir;

  public DiskBackedLanczosState(VectorIterable corpus, int numCols, int desiredRank,
      Vector initialVector, File dir) {
    super(corpus, numCols, desiredRank, initialVector);
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