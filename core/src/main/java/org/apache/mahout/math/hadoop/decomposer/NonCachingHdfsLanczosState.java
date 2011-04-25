package org.apache.mahout.math.hadoop.decomposer;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

import java.io.IOException;

/**
 * 
 */
public class NonCachingHdfsLanczosState extends HdfsBackedLanczosState {
  public NonCachingHdfsLanczosState(VectorIterable corpus, int numCols, int desiredRank,
      Vector initialVector, Path dir) {
    super(corpus, numCols, desiredRank, initialVector, dir);
  }

  @Override
  protected void updateHdfsState() throws IOException {
    super.updateHdfsState();
    basis.clear();
    singularVectors.clear();
  }
}
