package org.apache.mahout.clustering.lda.cvb;

public enum AggregationBranch {
  TOPIC_TERM, DOC_TOPIC, TOPIC_SUM;

  public static AggregationBranch of(int termId, int docId) {
    if(termId < 0 && docId < 0) {
      return TOPIC_SUM;
    }
    if(termId < 0) {
      return DOC_TOPIC;
    }
    if(docId < 0) {
      return TOPIC_TERM;
    }
    throw new UnsupportedOperationException("No branch for: " + termId + "," + docId);
  }
}
