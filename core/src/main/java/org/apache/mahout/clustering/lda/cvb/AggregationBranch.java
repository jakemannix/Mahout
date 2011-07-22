package org.apache.mahout.clustering.lda.cvb;

public enum AggregationBranch {
  TOPIC_TERM, DOC_TOPIC, TOPIC_SUM;

  public static AggregationBranch of(int termId, int docId) {
    return termId >= 0 ? TOPIC_TERM : docId >= 0 ? DOC_TOPIC : TOPIC_SUM;
  }
}
