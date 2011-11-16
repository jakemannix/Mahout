package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Charsets;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
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
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.vectorizer.encoders.MurmurHash;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

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
  Path otp = new Path(basePath, "otp.txt");

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
      List<String> perplexityStrings = Lists.newArrayList();
      List<Double> perplexities = Lists.newArrayList();
      double p0 = -1;
      while(i < 25) {
        double perplexity = model.perplexity(doc, docTopicCounts);
        model.trainDocTopicModel(doc, docTopicCounts, docTopicModel);
        if(p0 < 0) {
          p0 = perplexity;
        }
        double relativePerplexity = (1 - perplexity / p0);
        double previous = i == 0 ? relativePerplexity : perplexities.get(i - 1);
        perplexityStrings.add(String.format("%.6f", relativePerplexity - previous));
        perplexities.add(relativePerplexity);
        i++;
      }
      System.out.println("numUniqueTerms: " + doc.getNumNondefaultElements());
      Collections.sort(perplexityStrings, Collections.<String>reverseOrder());
      System.out.println(Joiner.on(", ").join(perplexityStrings));
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

  @Test
  public void testOverlappingTriangles() throws Exception {
    Matrix overlappingTriangles = loadOverlappingTrianglesCorpus();
    String[] terms = new String[26];
    for(int i=0; i<terms.length; i++) {
      terms[i] = "" + ((char)(i + 97));
    }
    InMemoryCollapsedVariationalBayes0 cvb =
        new InMemoryCollapsedVariationalBayes0(overlappingTriangles, terms, 5, 0.01, 0.01, 1, 1, 1);
    cvb.setVerbose(true);
    cvb.iterateUntilConvergence(0, 100, 10);
    cvb.writeModel(new Path("/tmp/topics"));
  }

  @Test
  public void testLoadTopics() throws Exception {
    String[] terms = new String[26];
    for(int i=0; i<terms.length; i++) {
      terms[i] = "" + ((char)(i + 97));
    }
    TopicModel otpModel = new TopicModel(new Configuration(), 0.01, 0.01, terms, 1, 1,
        new Path("/tmp/topics"));
    Matrix otpMatrix = otpModel.topicTermCounts();
    for(int topic = 0; topic < otpMatrix.numRows(); topic++) {
      otpMatrix.getRow(topic).assign(Functions.mult(1/otpMatrix.getRow(topic).norm(1)));
    }
    System.out.println(otpModel.toString());
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

  // {
  //   { 1, 2, 4, 2, 1, 0, 0, 0, 0, 0, 0 },
  //   { 0, 0, 0, 1, 2, 4, 2, 1, 0, 0, 0 },
  //   { 0, 0, 0, 0, 0, 0, 1, 2, 4, 2, 1 }
  // }  numTopics * width = numTerms
  public static Matrix randomStructuredModel(int numTopics, int numTerms) {
    Matrix model = new DenseMatrix(numTopics, numTerms);
    int width = (numTerms + 1) / (numTopics + 1);
    for(int topic = 0; topic < numTopics; topic++) {
      int topicCentroid = width * (1+topic);
      int start = topicCentroid - width;
      for(int i = 0; i < width * 2 && start + i < model.numCols(); i++) {
        double v = 1.0 / (1 + Math.abs((i-1) - 0.5 * width));
        model.set(topic, start + i, v);
      }
    }
    return model;
  }

  @Test
  public void testRandomStructuredModel() throws Exception {
    Matrix model = randomStructuredModel(3, 11);

  }

  public Matrix loadOverlappingTrianglesCorpus() throws IOException {
    List<Vector> vectors = Lists.newArrayList(
        Collections2.transform(Collections2.filter(Files.readLines(
          new File("/Users/jake/open_src/gitrepo/mahout/examples/bin/work/20news-bydate/otp.txt"),
          Charsets.UTF_8),
        Predicates.contains(Pattern.compile("[a-z]"))), new Function<String, Vector>() {
      @Override public Vector apply(String s) {
        Vector vector = new DenseVector(26);
        String[] tokens = s.split(" ");
        for(String token : tokens) {
          char t = token.charAt(0);
          if(Character.isLetter(t) && Character.isLowerCase(t)) {
            vector.set(t - 97, vector.get(t - 97) + 1);
          }
        }
        return vector;
      }
    }));

    return new SparseRowMatrix(new int[] {vectors.size(), 26},
        vectors.toArray(new Vector[vectors.size()]));
  }
}
