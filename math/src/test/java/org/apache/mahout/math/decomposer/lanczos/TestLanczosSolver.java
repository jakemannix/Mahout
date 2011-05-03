/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.decomposer.lanczos;

import org.apache.mahout.math.AbstractMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.SolverTest;
import org.apache.mahout.math.matrix.DoubleMatrix1D;
import org.apache.mahout.math.matrix.linalg.EigenvalueDecomposition;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestLanczosSolver extends SolverTest {
  private static final Logger log = LoggerFactory.getLogger(TestLanczosSolver.class);

  private static final double ERROR_TOLERANCE = 1.0e-5;

  @Test
  public void testEigenvalueCheck() throws Exception {
    int size = 100;
    Matrix m = randomHierarchicalSymmetricMatrix(size);
    int desiredRank = 80;

    Vector initialVector = new DenseVector(size);
    initialVector.assign(1d / Math.sqrt(size));
    LanczosSolver solver = new LanczosSolver();
    LanczosState state = new LanczosState(m, size, true, desiredRank, initialVector);
    // set initial vector?
    solver.solve(state, desiredRank);

    EigenvalueDecomposition decomposition = new EigenvalueDecomposition(m);
    DoubleMatrix1D eigenvalues = decomposition.getRealEigenvalues();

    float fractionOfEigensExpectedGood = 0.75f;
    for(int i = 0; i < fractionOfEigensExpectedGood * desiredRank; i++) {
      log.info(i + " : L = {}, E = {}",
          state.getSingularValue(desiredRank - i - 1),
          eigenvalues.get(eigenvalues.size() - i - 1) );
      Vector v = state.getRightSingularVector(i);
      Vector v2 = decomposition.getV().viewColumn(eigenvalues.size() - i - 1).toVector();
      double error = 1 - Math.abs(v.dot(v2)/(v.norm(2) * v2.norm(2)));
      log.info("error: {}", error);
      assertTrue(i + ": 1 - cosAngle = " + error, error < ERROR_TOLERANCE);
    }
  }


  @Test
  public void testLanczosSolver() throws Exception {
    int numRows = 800;
    int numColumns = 500;
    Matrix corpus = randomHierarchicalMatrix(numRows, numColumns, false);
    int rank = 50;
    Vector initialVector = new DenseVector(numColumns);
    initialVector.assign(1d / Math.sqrt(numColumns));
    LanczosState state = new LanczosState(corpus, numColumns, false, rank, initialVector);
    long time = timeLanczos(corpus, state, rank);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    assertOrthonormal(state);
    for(int i = 0; i < rank/2; i++) {
      assertEigen(i, state.getRightSingularVector(i), corpus, ERROR_TOLERANCE, false);
    }
    //assertEigen(eigens, corpus, rank / 2, ERROR_TOLERANCE, false);
  }

  @Test
  public void testLanczosSolverSymmetric() throws Exception {
    int numCols = 500;
    Matrix corpus = randomHierarchicalSymmetricMatrix(numCols);
    int rank = 30;
    Vector initialVector = new DenseVector(numCols);
    initialVector.assign(1d / Math.sqrt(numCols));
    LanczosState state = new LanczosState(corpus, numCols, true, rank, initialVector);
    long time = timeLanczos(corpus, state, rank);
    assertTrue("Lanczos taking too long!  Are you in the debugger? :)", time < 10000);
    //assertOrthonormal(state);
    //assertEigen(state, rank / 2, ERROR_TOLERANCE, true);
  }

  public static long timeLanczos(Matrix corpus, LanczosState state, int rank) {
    long start = System.currentTimeMillis();

    LanczosSolver solver = new LanczosSolver();
    // initialize!
    solver.solve(state, rank);
    
    long end = System.currentTimeMillis();
    return end - start;
  }

}
