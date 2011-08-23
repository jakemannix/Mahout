package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class CVB0Combiner extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {


  private double[] topicSums;

  @Override
  protected void setup(Context context) {
    topicSums = new double[context.getConfiguration().getInt(CVB0Mapper.NUM_TOPICS, -1)];
  }

  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context ctx)
      throws IOException, InterruptedException {
    int termId = key.getTermId(); // a
    int docId = key.getDocId(); // i
    final boolean aggregationPass = key.isB();
    if(!aggregationPass) {
      // just pass through!
      for(CVBTuple tuple : values) {
        ctx.write(key, tuple);
      }
    } else {
      AggregationBranch branch = AggregationBranch.of(termId, docId);
      CVBTuple aggregateTuple = new CVBTuple();
      aggregateTuple.setCount(branch, new double[topicSums.length]);
      for(CVBTuple tuple : values) {
        for(AggregationBranch otherBranches : AggregationBranch.values()) {
          if(otherBranches != branch && tuple.hasData(otherBranches)) {
            // why do we have data on another branch?
            throw new IllegalStateException("wrong branch: " + key + " : " + otherBranches + " : " + tuple);
          }
        }
        if(tuple.getCount(branch) == null && tuple.getTopic() < 0) {
          throw new IllegalStateException("Should have data for branch: " + branch + " or at least"
                                          + " topic != " + tuple.getTopic());
        } else if (tuple.getTopic() >= 0) {
          aggregateTuple.sparseAccumulate(branch, tuple);
        } else if (tuple.getCount(branch) != null) {
          aggregateTuple.accumulate(branch, tuple);
        }
      }
      ctx.write(key, aggregateTuple);
    }
  }
}
