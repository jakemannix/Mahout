package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

public class CVBTopicOutputMapper extends CVB0Mapper {

  @Override
  public void map(CVBKey key, CVBTuple value, Context context) throws IOException,
      InterruptedException {
    double[] t_ax = value.getCount(AggregationBranch.TOPIC_TERM);
    double[] t_x = value.getCount(AggregationBranch.TOPIC_SUM);
    double[] pTopicGivenTerm;
    // p(a | x) = p(x | a) * p(a)/p(x)

  }
}
