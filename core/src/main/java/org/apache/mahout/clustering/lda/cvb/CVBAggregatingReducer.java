package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class CVBAggregatingReducer extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {

  @Override
  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context context)
      throws IOException, InterruptedException {
    CVBTuple outputTuple = new CVBTuple();
    for(CVBTuple tuple : values) {
      for(AggregationBranch branch : AggregationBranch.values()) {
        double[] counts = tuple.getCount(branch);
        if(counts != null) {
          outputTuple.setCount(branch, counts);
        }
      }
    }
    context.write(key, outputTuple);
  }
}
