package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class TermDedupingMapper extends Mapper<CVBKey, CVBTuple, CVBKey, CVBTuple> {
  @Override
  public void map(CVBKey key, CVBTuple value, Context context)
      throws IOException, InterruptedException {
    key.setDocId(-1);
    key.setBranch(AggregationBranch.of(key.getTermId(), key.getDocId()));
    context.write(key, value);
  }
}
