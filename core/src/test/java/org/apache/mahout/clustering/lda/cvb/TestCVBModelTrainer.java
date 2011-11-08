package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;
import junit.framework.TestCase;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.MatrixUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.MurmurHash;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class TestCVBModelTrainer extends TestCase {

  double eta = 0.001;
  double alpha = 0.001;
  String[] dictionary = null;
  int numThreads = 1;
  double modelWeight = 1d;
  Path basePath = new Path("/Users/jake/open_src/gitrepo/mahout/examples/bin/work/20news-bydate");
  Path modelPath = new Path(basePath, "topics_2");
  Path corpusPath = new Path(basePath, "vectorized/int_vectors/matrix");
  Path dictionaryPath = new Path(basePath, "vectorized/dictionary.file-0");

  TopicModel model;
  Matrix corpus;

  @Before
  public void setUp() throws Exception {
    Configuration conf = new Configuration();
    dictionary = MatrixUtils.invertDictionary(MatrixUtils.readDictionary(conf, dictionaryPath));
    model = new TopicModel(conf, eta, alpha, dictionary, numThreads,
        modelWeight, modelPath);
    corpus = MatrixUtils.read(conf, corpusPath);
  }

  @Test
  public void testSingleDocumentConvergence() throws Exception {
    int j = 0;
    for(MatrixSlice slice : corpus) {
      Vector doc = slice.vector();
      assertNotNull(doc);
      System.out.println(doc.asFormatString(dictionary));
      Vector docTopicCounts = new DenseVector(model.getNumTopics());
      docTopicCounts.assign(1/model.getNumTopics());
      Matrix docTopicModel =
          new SparseRowMatrix(new int[] {model.getNumTopics(), model.getNumTerms()}, true);
      int i = 0;
      List<String> perplexities = Lists.newArrayList();
      double p0 = -1;
      while(i < 25) {
        double perplexity = model.perplexity(doc, docTopicCounts);
        model.trainDocTopicModel(doc, docTopicCounts, docTopicModel);
        if(p0 < 0) {
          p0 = perplexity;
        }
        perplexities.add(String.format("%.6f", (1 - perplexity / p0)));
        i++;
      }
      System.out.println("numUniqueTerms: " + doc.getNumNondefaultElements());
      System.out.println(Joiner.on(", ").join(perplexities));
      if(j++ > 100) break;
    }
  }

  @Test
  public void testPerplexityVariance() throws Exception {
    model = new TopicModel(model.getNumTopics(), model.getNumTerms(), eta, alpha, new Random(1234L),
       dictionary, 1, 1);
    TopicModel updatedModel = new TopicModel(model.getNumTopics(), model.getNumTerms(), eta, alpha,
        dictionary, 1, 1);
    Matrix docTopicCounts = new DenseMatrix(corpus.numRows(), model.getNumTopics());
    docTopicCounts.assign(1/model.getNumTopics());
    double startingPerplexity = -1;
    double previousPerplexity = -1;
    for(int i = 0; i < 100; i++) {
      List<Pair<Double,Double>> perplexitiesWithWeights = Lists.newArrayList();
      for(int docId = 0; docId < corpus.numRows(); docId++) {
        perplexitiesWithWeights.add(
            Pair.of(model.perplexity(corpus.getRow(docId), docTopicCounts.getRow(docId)),
                    corpus.getRow(docId).norm(1)));
        Matrix docTopicModel
            = new SparseRowMatrix(new int[] {model.getNumTopics(), model.getNumTerms()}, true);
        model.trainDocTopicModel(corpus.getRow(docId), docTopicCounts.getRow(docId), docTopicModel);
        for(int t = 0; t < docTopicModel.numRows(); t++) {
          updatedModel.updateTopic(t, docTopicModel.getRow(t));
        }
      }
      model = updatedModel;
      updatedModel = new TopicModel(model.getNumTopics(), model.getNumTerms(), eta, alpha,
        dictionary, 1, 1);
      double perplexity = sum(perplexitiesWithWeights, true) / sum(perplexitiesWithWeights, false);
      if(startingPerplexity < 0) {
        startingPerplexity = perplexity;
        previousPerplexity = perplexity;
      }
      double delta = (perplexity - previousPerplexity) / startingPerplexity;
      System.out.println(delta + " : cumulative delta:" + (1 - perplexity / startingPerplexity)
                         + ", Perplexity: " + perplexity);
      previousPerplexity = perplexity;
    }
  }

  private static Vector hash(Vector v, int dim, int numProbes, int seed) {
    Vector hashedVector = new RandomAccessSparseVector(dim, v.getNumNondefaultElements() * numProbes);
    Iterator<Vector.Element> it = v.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      for(int probe = 0; probe < numProbes; probe++) {
        int hashedFeature = (MurmurHash.hash(Ints.toByteArray(e.index() ^ (probe * seed)), seed)
                             & Integer.MAX_VALUE) % dim;
        hashedVector.set(hashedFeature, hashedVector.get(hashedFeature) + e.get() / numProbes);
      }
    }
    return hashedVector;
  }

  @Test
  public void testTrainHashedModel() throws Exception {
    int hashedFeatureDim = model.getNumTerms() / 10;
    int numProbes = 4;
    int seed = 1234;
    int numTopics = model.getNumTopics();
    TopicModel hashedModel = new TopicModel(numTopics, hashedFeatureDim, eta, alpha,
        new Random(1234L), dictionary, 1, 1);
    TopicModel updatedModel = new TopicModel(numTopics, hashedFeatureDim, eta, alpha,
        dictionary, 1, 1);
    Matrix docTopicCounts = new DenseMatrix(corpus.numRows(), numTopics);
    double startingPerplexity = -1;
    double previousPerplexity = -1;
    for(int iteration = 0; iteration < 100; iteration++) {
      List<Pair<Double,Double>> perplexitiesWithWeights = Lists.newArrayList();
      for(int docId = 0; docId < corpus.numRows(); docId++) {
        Vector originalDocument = corpus.getRow(docId);
        Vector hashedVector = hash(originalDocument, hashedFeatureDim, numProbes, seed);
        perplexitiesWithWeights.add(
            Pair.of(hashedModel.perplexity(hashedVector, docTopicCounts.getRow(docId)),
                    hashedVector.norm(1)));
        Matrix docTopicModel
            = new SparseRowMatrix(new int[] {numTopics, hashedFeatureDim}, true);
        hashedModel.trainDocTopicModel(hashedVector, docTopicCounts.getRow(docId), docTopicModel);
        for(int t = 0; t < docTopicModel.numRows(); t++) {
          updatedModel.updateTopic(t, docTopicModel.getRow(t));
        }
      }
      hashedModel = updatedModel;
      updatedModel = new TopicModel(numTopics, hashedFeatureDim, eta, alpha,
        dictionary, 1, 1);
      double perplexity = sum(perplexitiesWithWeights, true) / sum(perplexitiesWithWeights, false);
      if(startingPerplexity < 0) {
        startingPerplexity = perplexity;
        previousPerplexity = perplexity;
      }
      double delta = (perplexity - previousPerplexity) / startingPerplexity;
      System.out.println(delta + " : cumulative delta:" + (1 - perplexity / startingPerplexity)
                         + ", Perplexity: " + perplexity);
      previousPerplexity = perplexity;
    }
    hashedModel.setConf(new Configuration());
    hashedModel.persist(new Path(basePath, "hashed-" + hashedFeatureDim), true);
  }

  public double klDivergence(Matrix p, Matrix q) {
    int numTopics = p.numRows();
    double divergence = 0;
    for(int topic = 0; topic < numTopics; topic++) {
      Vector pv = p.getRow(topic).normalize(1);
      Vector qv = q.getRow(topic).normalize(1);
      for(Vector.Element e : qv) {
        if(e.get() > 0 && pv.get(e.index()) > 0) {
          divergence += pv.get(e.index()) * Math.log(pv.get(e.index()) / e.get());
        }
      }
    }
    return divergence;
  }

  public static double sum(List<Pair<Double,Double>> list, boolean first) {
    double sum = 0;
    for(Pair<Double,Double> p : list) {
      sum += (first ? p.getFirst() : p.getSecond());
    }
    return sum;
  }

}
