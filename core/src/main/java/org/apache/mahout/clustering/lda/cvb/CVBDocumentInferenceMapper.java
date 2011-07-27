package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

public class CVBDocumentInferenceMapper extends CVB0Mapper {

  @Override
  public void map(CVBKey key, CVBTuple value, Context context) throws IOException,
      InterruptedException {
    double[] pTopicGivenDoc = value.getCount(AggregationBranch.DOC_TOPIC);
    double total = 0;
    for(double d : pTopicGivenDoc) {
      total += d;
    }
    for(int x = 0; x < pTopicGivenDoc.length; x++) {
      pTopicGivenDoc[x] /= total;
    }
    value.setCount(AggregationBranch.TOPIC_TERM, null);
    value.setCount(AggregationBranch.TOPIC_SUM, null);
    key.setTermId(-1);
    context.write(key, value);
  }
}
