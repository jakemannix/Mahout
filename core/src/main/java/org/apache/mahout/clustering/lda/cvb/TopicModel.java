package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class TopicModel {
  private final String[] dictionary;
  private final Matrix topicTermCounts;
  private final Vector topicSums;
  private final int numTopics;
  private final int numTerms;
  private final double eta;
  private final double alpha;

  private Sampler sampler;

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, Random random,
      String[] dictionary) {
    this.dictionary = dictionary;
    this.numTopics = numTopics;
    this.numTerms = numTerms;
    this.eta = eta;
    this.alpha = alpha;
    this.sampler = new Sampler(random);
    this.topicTermCounts = new SparseRowMatrix(new int[]{numTopics, numTerms}, true);
    this.topicSums = new DenseVector(numTopics);
    for(int x = 0; x < numTopics; x++) {
      for(int term = 0; term < numTerms; term++) {
        topicTermCounts.getRow(x).set(term, random.nextDouble());
      }
      topicSums.set(x, topicTermCounts.getRow(x).norm(1));
    }
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    String[] dictionary) {
    this.dictionary = dictionary;
    this.topicTermCounts = topicTermCounts;
    this.topicSums = topicSums;
    this.numTopics = topicSums.size();
    this.numTerms = topicTermCounts.numCols();
    this.eta = eta;
    this.alpha = alpha;
    this.sampler = new Sampler(new Random(1234));
  }

  public String toString() {
    String buf = "";
    for(int x = 0; x < numTopics; x++) {
      buf += vectorToSortedString(topicTermCounts.getRow(x), dictionary).substring(0, 1000) + "\n";
    }
    return buf;
  }

  public int sampleTerm(Vector topicDistribution) {
    return sampler.sample(topicTermCounts.getRow(sampler.sample(topicDistribution)));
  }

  public int sampleTerm(int topic) {
    return sampler.sample(topicTermCounts.getRow(topic));
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
      docTopicCounts.getRow(x).addTo(topicTermCounts.getRow(x));
      topicSums.set(x, topicSums.get(x) + docTopicCounts.getRow(x).norm(1));
    }
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
    while(listIt.hasNext()) {
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

}
