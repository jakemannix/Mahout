package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DistributedRowMatrixWriter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class TopicModel implements Configurable {
  private static final Logger log = LoggerFactory.getLogger(TopicModel.class);
  private final String[] dictionary;
  private final Matrix topicTermCounts;
  private final Vector topicSums;
  private final int numTopics;
  private final int numTerms;
  private final double eta;
  private final double alpha;

  private Configuration conf;

  private Sampler sampler;
  private final int numThreads;
  private ThreadPoolExecutor threadPool = new ThreadPoolExecutor(32, 32, Integer.MAX_VALUE,
      TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(100));
  private Updater[] updaters;

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, String[] dictionary) {
    this(numTopics, numTerms, eta, alpha, null, dictionary, 1);
  }

  public TopicModel(Path modelPath, Configuration conf, double eta, double alpha,
      String[] dictionary, int numThreads) throws IOException {
    this(loadModel(modelPath, conf), eta, alpha, dictionary, numThreads);
  }

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, Random random,
      String[] dictionary, int numThreads) {
    this(randomMatrix(numTopics, numTerms, random), eta, alpha, dictionary, numThreads);
  }

  private TopicModel(Pair<Matrix, Vector> model, double eta, double alpha, String[] dict,
      int numThreads) {
    this(model.getFirst(), model.getSecond(), eta, alpha, dict, numThreads);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    String[] dictionary) {
    this(topicTermCounts, topicSums, eta, alpha, dictionary, 1);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    String[] dictionary, int numThreads) {
    this.dictionary = dictionary;
    this.topicTermCounts = topicTermCounts;
    this.topicSums = topicSums;
    this.numTopics = topicSums.size();
    this.numTerms = topicTermCounts.numCols();
    this.eta = eta;
    this.alpha = alpha;
    this.sampler = new Sampler(new Random(1234));
    this.numThreads = numThreads;
    initializeThreadPool();
  }

  private void initializeThreadPool() {
    threadPool = new ThreadPoolExecutor(numThreads, numThreads, 0, TimeUnit.SECONDS,
        new ArrayBlockingQueue<Runnable>(numThreads));
    threadPool.allowCoreThreadTimeOut(false);
    updaters = new Updater[numThreads];
    for(int i = 0; i < numThreads; i++) {
      updaters[i] = new Updater();
      threadPool.submit(updaters[i]);
    }

  }

  private static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, Random random) {
    Matrix topicTermCounts = new SparseRowMatrix(new int[]{numTopics, numTerms}, true);
    Vector topicSums = new DenseVector(numTopics);
    if(random != null) {
      for(int x = 0; x < numTopics; x++) {
        for(int term = 0; term < numTerms; term++) {
          topicTermCounts.getRow(x).set(term, random.nextDouble());
        }
      }
    }
    for(int x = 0; x < numTopics; x++) {
      topicSums.set(x, random == null ? 1d : topicTermCounts.getRow(x).norm(1));
    }
    return Pair.of(topicTermCounts, topicSums);
  }

  public static Pair<Matrix, Vector> loadModel(Path modelPath, Configuration conf)
      throws IOException {
    List<Pair<Integer, Vector>> rows = Lists.newArrayList();
    for(Pair<IntWritable, VectorWritable> row :
        new SequenceFileIterable<IntWritable, VectorWritable>(modelPath, true, conf)) {
      rows.add(Pair.of(row.getFirst().get(), row.getSecond().get()));
    }
    if(rows.isEmpty()) {
      throw new IOException(modelPath + " has no vectors in it");
    }
    Vector[] v = new Vector[rows.size()];
    Vector topicSums = new DenseVector(v.length);
    for(Pair<Integer, Vector> pair : rows) {
      v[pair.getFirst()] = pair.getSecond();
      topicSums.set(pair.getFirst(), pair.getSecond().norm(1));
    }
    Matrix model = new SparseRowMatrix(new int[]{v.length, v[0].size()},
        v, true, v[0] instanceof SequentialAccessSparseVector);
    return Pair.of(model, topicSums);
  }

  public String toString() {
    String buf = "";
    for(int x = 0; x < numTopics; x++) {
      String v = vectorToSortedString(topicTermCounts.getRow(x), dictionary);
      buf += v + "\n";
    }
    return buf;
  }

  public int sampleTerm(Vector topicDistribution) {
    return sampler.sample(topicTermCounts.getRow(sampler.sample(topicDistribution)));
  }

  public int sampleTerm(int topic) {
    return sampler.sample(topicTermCounts.getRow(topic));
  }

  public void reset() {
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, new SequentialAccessSparseVector(numTerms));
    }
    topicSums.assign(1d);
    initializeThreadPool();
  }

  public void awaitTermination() {
    for(Updater updater : updaters) {
      updater.shutdown();
    }
  }

  public void renormalize() {
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, topicTermCounts.getRow(x).normalize(1));
      topicSums.assign(1d);
    }
  }

  public void trainDocTopicModel(Vector original, Vector topics, Matrix docTopicModel) {
    // first calculate p(topic|term,document) for all terms in original, and all topics,
    // using p(term|topic) and p(topic|doc)
    pTopicGivenTerm(original, topics, docTopicModel);
    normalizeByTopic(docTopicModel);
    // now multiply, term-by-term, by the document, to get the weighted distribution of
    // term-topic pairs from this document.
    for(int x = 0; x < numTopics; x++) {
      docTopicModel.assignRow(x, docTopicModel.getRow(x).times(original));
    }
    // now recalculate p(topic|doc) by summing contributions from all of pTopicGivenTerm
    topics.assign(0d);
    for(int x = 0; x < numTopics; x++) {
      topics.set(x, docTopicModel.getRow(x).norm(1));
    }
    // now renormalize so that sum_x(p(x|doc)) = 1
    topics.assign(Functions.mult(1/topics.norm(1)));
  }

  public Vector infer(Vector original, Vector docTopics) {
    Vector pTerm = original.like();
    Iterator<Vector.Element> it = original.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int term = e.index();
      // p(a) = sum_x (p(a|x) * p(x|i))
      double pA = 0;
      for(int x = 0; x < numTopics; x++) {
        pA += (topicTermCounts.getRow(x).get(term) / topicSums.get(x)) * docTopics.get(x);
      }
      pTerm.set(term, pA);
    }
    return pTerm;
  }

  public void update(Matrix docTopicCounts) {
    for(int x = 0; x < numTopics; x++) {
      updaters[x % updaters.length].update(x, docTopicCounts.getRow(x));
    }
  }

  private void updateTopic(int topic, Vector docTopicCounts) {
    docTopicCounts.addTo(topicTermCounts.getRow(topic));
    topicSums.set(topic, topicSums.get(topic) + docTopicCounts.norm(1));
  }

  public void update(int termId, Vector topicCounts) {
    for(int x = 0; x < numTopics; x++) {
      Vector v = topicTermCounts.getRow(x);
      v.set(termId, v.get(termId) + topicCounts.get(x));
    }
    topicCounts.addTo(topicSums);
  }

  public void persist(Path outputDir, boolean overwrite) throws IOException {
    FileSystem fs = outputDir.getFileSystem(conf);
    if(overwrite) {
      fs.delete(outputDir, true); // CHECK second arg
    }
    DistributedRowMatrixWriter.write(outputDir, conf, topicTermCounts);
  }

  /**
   *
   * @param docTopics d[x] is the overall weight of topic_x in this document.
   * @return pTGT[x].get(a) is the (un-normalized) p(x|a,i), or if docTopics is null,
   * p(a|x) (also un-normalized)
   */
  private void pTopicGivenTerm(Vector document, Vector docTopics, Matrix termTopicDist) {
    for(int x = 0; x < numTopics; x++) {
      Iterator<Vector.Element> it = document.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        int term = e.index();
        double d = docTopics == null ? 1d : docTopics.get(x);
        double p = (topicTermCounts.getRow(x).get(term) + eta) * (d + alpha) / (topicSums.get(x) + eta * numTerms);
        termTopicDist.getRow(x).set(e.index(), p);
      }
    }
  }

  private void normalizeByTopic(Matrix perTopicSparseDistributions) {
    Iterator<Vector.Element> it = perTopicSparseDistributions.getRow(0).iterateNonZero();
    // then make sure that each of these is properly normalized by topic: sum_x(p(x|t,d)) = 1
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int a = e.index();
      double sum = 0;
      for(int x = 0; x < numTopics; x++) {
        sum += perTopicSparseDistributions.getRow(x).get(a);
      }
      for(int x = 0; x < numTopics; x++) {
        perTopicSparseDistributions.getRow(x).set(a,
            perTopicSparseDistributions.getRow(x).get(a) / sum);
      }
    }
  }

  public static String vectorToSortedString(Vector vector, String[] dictionary) {
    List<Pair<String,Double>> vectorValues =
        new ArrayList<Pair<String, Double>>(vector.getNumNondefaultElements());
    Iterator<Vector.Element> it = vector.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      vectorValues.add(Pair.of(dictionary != null ? dictionary[e.index()] : String.valueOf(e.index()),
                               e.get()));
    }
    Collections.sort(vectorValues, new Comparator<Pair<String, Double>>() {
      @Override public int compare(Pair<String, Double> x, Pair<String, Double> y) {
        return y.getSecond().compareTo(x.getSecond());
      }
    });
    Iterator<Pair<String,Double>> listIt = vectorValues.iterator();
    StringBuilder bldr = new StringBuilder(2048);
    bldr.append("{");
    int i = 0;
    while(listIt.hasNext() && i < 25) {
      i++;
      Pair<String,Double> p = listIt.next();
      bldr.append(p.getFirst());
      bldr.append(":");
      bldr.append(p.getSecond());
      bldr.append(",");
    }
    if(bldr.length() > 1) {
      bldr.setCharAt(bldr.length() - 1, '}');
    }
    return bldr.toString();
  }

  @Override
  public void setConf(Configuration configuration) {
    this.conf = configuration;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  private final class Updater implements Runnable {
    private ArrayBlockingQueue<Pair<Integer, Vector>> queue =
        new ArrayBlockingQueue<Pair<Integer, Vector>>(100);
    private boolean shutdown = false;
    private boolean shutdownComplete = false;

    public void shutdown() {
      try {
        synchronized (this) {
          while(!shutdownComplete) {
            shutdown = true;
            wait();
          }
        }
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to shutdown() : ", e);
      }
    }

    public boolean update(int topic, Vector v) {
      if(shutdown) { // maybe don't do this?
        throw new IllegalStateException("In SHUTDOWN state: cannot submit tasks");
      }
      while(true) { // keep trying if interrupted
        try {
          // start async operation by submitting to the queue
          queue.put(Pair.of(topic, v));
          // return once you got access to the queue
          return true;
        } catch (InterruptedException e) {
          log.warn("Interrupted trying to queue update:", e);
        }
      }
    }

    @Override public void run() {
      while(!shutdown) {
        try {
          Pair<Integer, Vector> pair = queue.poll(1, TimeUnit.SECONDS);
          if(pair != null) {
            updateTopic(pair.getFirst(), pair.getSecond());
          }
        } catch (InterruptedException e) {
          log.warn("Interrupted waiting to poll for update", e);
        }
      }
      // in shutdown mode, finish remaining tasks!
      for(Pair<Integer, Vector> pair : queue) {
        updateTopic(pair.getFirst(), pair.getSecond());
      }
      synchronized (this) {
        shutdownComplete = true;
        notifyAll();
      }
    }
  }

}
