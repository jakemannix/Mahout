package org.apache.mahout.math.hadoop.decomposer;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.LinearOperator;
import org.apache.mahout.math.Vector;

/**
 * 
 */
public class NonCachingHdfsLanczosState extends HdfsBackedLanczosState {
  public NonCachingHdfsLanczosState(LinearOperator corpus, int numCols, boolean isSymmetric, int desiredRank,
      Vector initialVector, Path dir) {
    super(corpus, numCols, isSymmetric, desiredRank, initialVector, dir);
  }

  @Override
  protected void updateHdfsState() throws IOException {
    super.updateHdfsState();
    basis.clear();
    singularVectors.clear();
  }
}
