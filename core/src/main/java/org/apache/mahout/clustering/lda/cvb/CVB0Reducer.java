package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/* a bunch of possibilities for key:
  (a, -1, T) : (-, [t_aix], -, -)
  (-1, i, T) : (-, -, [t_aix], -)
  (-1, -1, T): (-, -, -, [t_aix])
  and for any of these:
  (-1,-1, F) : (c_ai, -, -, -)
  (a, -1, F) : (c_ai, -, -, -)
  (-1, i, F) : (c_ai, -, -, -)

  e.g. (a, -1, -1) : [ (-, [t_aix], -, -), (-, [t_aix], -, -), ...]
  another call: (a, -1, 1) : [ (c_a0, -, -, -), (c_a1, -, -, -), ...]
*/
public class CVB0Reducer extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {

  private double[] topicSums = null;

  @Override
  protected void setup(Context context) {
    topicSums = new double[context.getConfiguration().getInt(CVB0Mapper.NUM_TOPICS, -1)];
  }

  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context ctx)
      throws IOException, InterruptedException {
    int termId = key.getTermId(); // a
    int docId = key.getDocId(); // i
    AggregationBranch branch = AggregationBranch.of(termId, docId);
    CVBTuple aggregateTuple = new CVBTuple();
    aggregateTuple.setCount(branch, new double[topicSums.length]);
    for(CVBTuple tuple : values) {
      if(tuple.getTopic() >= 0 && tuple.getCount() >= 0) {
        // still aggregating counts
        aggregateTuple.sparseAccumulate(branch, tuple);
      } else if(tuple.hasData(branch)) {
        aggregateTuple.accumulate(branch, tuple);
      } else {
        // done aggregating, tag corpus entry and re-emit
        key.setDocId(tuple.getDocumentId());
        key.setTermId(tuple.getTermId());
        key.setB(true);
        key.setBranch(null);
        // tag
        if(aggregateTuple.hasData(branch)) {
          tuple.clearCounts();
          tuple.setCount(branch, aggregateTuple.getCount(branch));
          if(branch == AggregationBranch.TOPIC_TERM) {
            // we should also have topicSums here
            if(topicSums == null) {
              throw new IllegalArgumentException("We should have gotten topicSums already!");
            }
            // tag with topicSums
            // TODO: take the ratios right here! topicTerm + eta / topicSum + W*eta
            tuple.setCount(AggregationBranch.TOPIC_SUM, topicSums);
          }
        } else {
          throw new IllegalStateException("we're taggin' and baggin, but nothing to tag with!");
        }
        // we're tagging corpus entries, so we write here.
        write(key, tuple, ctx, branch);
      }
    }
    if(branch == AggregationBranch.TOPIC_SUM) {
      topicSums = aggregateTuple.getCount(branch);
    }
  }

  private void write(CVBKey key, CVBTuple tuple, Context ctx, AggregationBranch b)
      throws IOException, InterruptedException {
    if(!tuple.hasData(b)) {
      throw new IllegalArgumentException("no data for branch we were tagging with");
    }
    ctx.write(key, tuple);
  }
}
