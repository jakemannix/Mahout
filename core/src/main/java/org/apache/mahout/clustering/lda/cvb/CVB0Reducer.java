package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/* a bunch of possibilities for key:
  (a, -1, -1) : (-, [t_aix], -, -)
  (-1, i, -1) : (-, -, [t_aix], -)
  (-1, -1, -1): (-, -, -, [t_aix])
  and for any of these:
  (-1,-1, 1) : (c_ai, -, -, -)
  (a, -1, 1) : (c_ai, -, -, -)
  (-1, i, 1) : (c_ai, -, -, -)

  e.g. (a, -1, -1) : [ (-, [t_aix], -, -), (-, [t_aix], -, -), ...]
  another call: (a, -1, 1) : [ (c_a0, -, -, -), (c_a1, -, -, -), ...]
*/
public class CVB0Reducer extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {
  private int numTopics;

  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context ctx)
      throws IOException, InterruptedException {
    int termId = key.getTermId(); // a
    int docId = key.getDocId(); // i
    AggregationBranch branch = AggregationBranch.of(termId, docId);

    CVBTuple aggregateTuple = new CVBTuple();
    aggregateTuple.setCount(branch, new double[numTopics]);
    for(CVBTuple tuple : values) {
      if(tuple.getCount(branch) != null) {
        // still aggregating counts
        aggregateTuple.accumulate(branch, tuple);
      } else {
        // done aggregating, tag corpus entry and re-emit
        key.setDocId(tuple.getDocumentId());
        key.setTermId(tuple.getTermId());
        tuple.setCount(branch, aggregateTuple.getCount(branch));
        ctx.write(key, tuple);
      }
    }
  }
}
