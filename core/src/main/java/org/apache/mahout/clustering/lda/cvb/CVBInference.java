package org.apache.mahout.clustering.lda.cvb;

public class CVBInference {
  private final double eta;
  private final double alpha;
  private final double etaTimesNumTerms;

  public CVBInference(double eta, double alpha, int numTerms) {
    this.eta = eta;
    this.alpha = alpha;
    this.etaTimesNumTerms = eta * numTerms;
  }

  public double[] pTopicGivenTermInDoc(CVBTuple value) {
    double[] topicTermCounts = value.getCount(AggregationBranch.TOPIC_TERM);
    double[] topicCounts = value.getCount(AggregationBranch.TOPIC_SUM);
    double[] docTopicCounts = value.getCount(AggregationBranch.DOC_TOPIC);
    int numTopics = docTopicCounts.length;
    double[] d = new double[numTopics];
    double total = 0;
    for(int x = 0; x < numTopics; x++) {
    // p(x | a, i) =~ ((t_ax + eta)/(t_x + eta*W)) * (d_ix + alpha)
      d[x] = (topicTermCounts[x] + eta) / (topicCounts[x] + etaTimesNumTerms);
      d[x] *= (docTopicCounts[x] + alpha);
      total += d[x];
    }
    // L_1 normalize, to get p(x|a,i)
    for(int x = 0; x < numTopics; x++) {
      d[x] /= total;
    }
    return d;
  }

}
