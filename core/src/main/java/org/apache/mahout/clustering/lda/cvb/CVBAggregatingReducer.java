package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class CVBAggregatingReducer extends Reducer<CVBKey, CVBTuple, CVBKey, CVBTuple> {
  private static final Logger log = LoggerFactory.getLogger(CVBAggregatingReducer.class);

  @Override
  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context context)
      throws IOException, InterruptedException {
    CVBTuple outputTuple = new CVBTuple();
    int numTuplesForKey = 0;
    List<CVBTuple> all = Lists.newArrayList();
    boolean BLOW_UP = false;
    for(CVBTuple tuple : values) {
      all.add(new CVBTuple(tuple));
      numTuplesForKey++;
      for(AggregationBranch branch : AggregationBranch.values()) {
        double[] counts = tuple.getCount(branch);
        if(counts != null) {
          if(outputTuple.hasData(branch)) {
            //throw new IllegalStateException(outputTuple + " already has " + branch);
            BLOW_UP = true;
          } else {
            outputTuple.setCount(branch, counts);
          }
        }
      }
      // outputTuple starts empty, so fill it up:
      if(outputTuple.getItemCount() < 0) {
        outputTuple.setItemCount(tuple.getItemCount());
        outputTuple.setTermId(tuple.getTermId());
        outputTuple.setDocumentId(tuple.getDocumentId());
      } else {
        if(outputTuple.getDocumentId() != tuple.getDocumentId() &&
           outputTuple.getTermId() != tuple.getTermId()) {
          throw new IllegalArgumentException("tuples in this reducer have different termId/docId!"
            + outputTuple + " vs " + tuple);
        }
      }
    }
    if(BLOW_UP) {
      throw new IllegalStateException(Joiner.on("\n").join(all));
    }
    if(numTuplesForKey != AggregationBranch.values().length) {
      throw new IllegalArgumentException("Key has wrong #tuples: " + numTuplesForKey + ", "
                                         + key + " => " + outputTuple);
    }
    write(key, outputTuple, context);
  }

  private void write(CVBKey key, CVBTuple outputTuple, Context context)
      throws IOException, InterruptedException {
    context.write(key, outputTuple);
  }
}
