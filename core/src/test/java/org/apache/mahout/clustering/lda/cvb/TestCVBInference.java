package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public class TestCVBInference extends MahoutTestCase {

  private Random random;
  private int numTopics = 10;
  private int vocabSize = 1000;
  private TopicModel topicModel;

  @Before
  public void setup() {
    random = new Random(1234);
    topicModel = generateRandomModel(numTopics, vocabSize);
  }

  @Test
  public void testIterations() throws Exception {
    Vector docTopicDist = generateRandomDistribution();
    Vector document = randomVector(docTopicDist, 100);
    Vector trainedDocTopicDist = new DenseVector(numTopics);
    trainedDocTopicDist.assign(1d / numTopics);
    Matrix docTopicModel = new SparseRowMatrix(new int[]{numTopics, vocabSize}, true);
    int iteration = 0;
    double convergence = 1e-6;
    double currentConvergence = Double.MAX_VALUE;
    while(iteration < 1000 && currentConvergence > convergence) {
      Vector oldTopics = new DenseVector(trainedDocTopicDist);
      topicModel.trainDocTopicModel(document, trainedDocTopicDist, docTopicModel);
      currentConvergence = oldTopics.minus(trainedDocTopicDist).norm(1);
      iteration++;
    }
    assertTrue(currentConvergence < convergence);
  }

  @Test
  public void testInference() throws Exception {
    Vector docTopicDist = generateRandomDistribution();
    Vector document = randomVector(docTopicDist, 100);
    Vector trainedDocTopicDist = new DenseVector(numTopics);
    trainedDocTopicDist.assign(1d / numTopics);
    Matrix docTopicModel = new SparseRowMatrix(new int[]{numTopics, vocabSize}, true);
    int iteration = 0;
    double convergence = 1e-6;
    double currentConvergence = Double.MAX_VALUE;
    Vector inferredDoc = null;
    while(iteration < 1000 && currentConvergence > convergence) {
      topicModel.trainDocTopicModel(document, trainedDocTopicDist, docTopicModel);
      inferredDoc = topicModel.infer(document, trainedDocTopicDist);
      double inferredNorm = inferredDoc.norm(1);
      inferredDoc = inferredDoc.times(document.norm(1) / inferredNorm);
      currentConvergence = inferredDoc.normalize(1).minus(document.normalize(1)).norm(1);
      iteration++;
    }
    inferredDoc = topicModel.infer(document, trainedDocTopicDist);
    assertTrue(currentConvergence < convergence);
  }

  private Vector randomVector(Vector docTopicDist, int numSamples) {
    Vector document = new RandomAccessSparseVector(vocabSize);
    for(int s = 0; s < numSamples; s++) {
      int term = topicModel.sampleTerm(docTopicDist);
      document.set(term, document.get(term) + 1);
    }
    return document;
  }

  /**
   * @param numTopics
   * @param vocabSize
   * @return TopicModel
   */
  private TopicModel generateRandomModel(int numTopics, int vocabSize) {
    Matrix model = new SparseRowMatrix(new int[]{numTopics, vocabSize}, true);
    Vector topicSums = new DenseVector(numTopics);
    for(int x = 0; x < numTopics; x++) {
      int peakTerm = random.nextInt(vocabSize);
      for(int a = 0; a < vocabSize; a++) {
        model.getRow(x).set(a, random.nextDouble() / (1 + Math.abs(peakTerm - a)));
      }
      topicSums.set(x, model.getRow(x).norm(1));
    }
    return new TopicModel(model, topicSums, 0, 0, null);
  }

  private Vector generateRandomDistribution() {
    Vector dist = new DenseVector(numTopics);
    int peakTerm = random.nextInt(numTopics);
    for(int x = 0; x < numTopics; x++) {
      dist.set(x, random.nextDouble() / (1 + Math.abs(peakTerm - x)));
    }
    return dist.normalize(1);
  }

}
