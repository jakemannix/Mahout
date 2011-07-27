package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class CVB0Combiner extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {

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
      for(CVBTuple tuple : values) {
        for(AggregationBranch otherBranches : AggregationBranch.values()) {
          if(otherBranches != branch && tuple.hasData(otherBranches)) {
            // why do we have data on another branch?
            throw new IllegalStateException("wrong branch: " + key + " : " + otherBranches + " : " + tuple);
          }
        }
        if(tuple.getCount(branch) == null) {
          throw new IllegalStateException("Should have data for branch: " + branch);
        } else {
          aggregateTuple.accumulate(branch, tuple);
        }
      }
      ctx.write(key, aggregateTuple);
    }
  }
}
