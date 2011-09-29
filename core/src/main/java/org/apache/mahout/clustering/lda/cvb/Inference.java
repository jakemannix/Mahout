package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Iterator;

public class Inference {

  private final int k;
  private final double eta;
  private final double alpha;
  private final int vocabSize;

  public Inference(int numTopics, double eta, double alpha, int vocabSize) {
    this.k = numTopics;
    this.eta = eta;
    this.alpha = alpha;
    this.vocabSize = vocabSize;
  }

  public Inference(int numTopics) {
    this(numTopics, 0, 0, 1);
  }

  public Vector infer(Vector original, Vector[] termTopicCounts,
      double[] topicSums, double[] docTopics) {
    // get p(a|x)
    Vector[] pTopicGivenTerm = pTermGivenTopic(termTopicCounts, topicSums);
    // now calculate p(a) for each term already known to be in the document:
    Vector pTerm = original.like();
    Iterator<Vector.Element> it = original.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int term = e.index();
      // p(a) = sum_x (p(a|x) * p(x|i))
      double pA = 0;
      for(int x = 0; x < k; x++) {
        pA += pTopicGivenTerm[x].get(term) * docTopics[x];
      }
      pTerm.set(term, pA);
    }
    return pTerm;
  }


  public Vector[] pTermGivenTopic(Vector[] termTopicCounts, double[] topicsSums) {
    return pTopicGivenTerm(termTopicCounts, topicsSums, null);
  }

  /**
   *
   * @param termTopicCounts t[x].get(a) has the weight of topic_x from term_a
   * @param topicSums t[x] is the overall weight of topic_x
   * @param docTopics d[x] is the overall weight of topic_x in this document.
   * @return pTGT[x].get(a) is the (un-normalized) p(x|a,i), or if docTopics is null,
   * p(a|x) (also un-normalized)
   */
  private Vector[] pTopicGivenTerm(Vector[] termTopicCounts,
      double[] topicSums, double[] docTopics) {
    Vector[] pTopicGivenTerm = new Vector[k];
    for(int x = 0; x < k; x++) {
      pTopicGivenTerm[x] = termTopicCounts[x].like();
      Iterator<Vector.Element> it = termTopicCounts[x].iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        double d = docTopics == null ? 1d : docTopics[x];
        pTopicGivenTerm[x].set(e.index(),
            (e.get() + eta) * (d + alpha) / (topicSums[x] + eta * vocabSize));
      }
    }
    return pTopicGivenTerm;
  }

  public double trainDocTopics(Vector original, double[][] termTopicCounts, double[] topicSums,
      double[] docTopics, Vector[] docTermTopicCounts) {
    Vector[] termTopicCountVectors = new Vector[termTopicCounts.length];
    for(int term = 0; term < termTopicCounts.length; term++) {
      termTopicCountVectors[term] = new DenseVector(termTopicCounts[term], true);
    }
    return trainDocTopics(original, termTopicCountVectors, topicSums, docTopics, docTermTopicCounts);
  }

  public double trainDocTopics(Vector original, Vector[] termTopicCounts,
      double[] topicSums, double[] docTopics, Vector[] docTermTopicCounts) {
    // hang onto old docTopics, to compare after training
    double[] oldDocTopics = docTopics.clone();
    // first calculate p(topic|term,document) for all terms in original, and all topics,
    // using p(term|topic) and p(topic|doc)
    Vector[] pTopicGivenTerm = pTopicGivenTerm(termTopicCounts, topicSums, docTopics);
    normalizeByTopic(pTopicGivenTerm);
    for(int x = 0; x < docTermTopicCounts.length; x++) {
      docTermTopicCounts[x] = pTopicGivenTerm[x].times(original);
    }
    // now recalculate p(topic|doc) by summing contributions from all of pTopicGivenTerm
    Iterator<Vector.Element> it = original.iterateNonZero();
    Arrays.fill(docTopics, 0d);
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int a = e.index();
      double v = e.get();
      for(int x = 0; x < k; x++) {
        // p(x|doc) += p(x|term,doc) * v(term)
        docTopics[x] += v * pTopicGivenTerm[x].get(a);
      }
    }
    // now renormalize so that sum_x(p(x|doc)) = 1
    double docSum = 0;
    for(double v : docTopics) {
      docSum += v;
    }
    for(int x = 0; x < k; x++) {
      docTopics[x] /= docSum;
    }
    return new DenseVector(oldDocTopics).minus(new DenseVector(docTopics)).norm(1);
  }

  private void normalizeByTopic(Vector[] perTopicSparseDistributions) {
    Iterator<Vector.Element> it = perTopicSparseDistributions[0].iterateNonZero();
    // then make sure that each of these is properly normalized by topic: sum_x(p(x|t,d)) = 1
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int a = e.index();
      double sum = 0;
      for(int x = 0; x < k; x++) {
        sum += perTopicSparseDistributions[x].get(a);
      }
      for(int x = 0; x < k; x++) {
        perTopicSparseDistributions[x].set(a, perTopicSparseDistributions[x].get(a) / sum);
      }
    }

  }
}
