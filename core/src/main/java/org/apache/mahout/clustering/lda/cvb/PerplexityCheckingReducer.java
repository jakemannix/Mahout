package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class PerplexityCheckingReducer extends Reducer<CVBKey, CVBTuple, NullWritable, DoubleWritable> {

  private double perplexity = 0;

  /*
   *  have p(a|x), d(x|i) for all a in i.
   *  normally compute p(x|a,i) = p(a|x)*d(x|i); p(x|a,i) /= sum_x(p(x|a,i))
   *  now instead want to properly normalize the d(x|i) by itself:
   *  compute p(x|i) = d(x|i) / sum_x(d(x|i))
   *  and compute sum_a,i(-log(sum_x(p(a|x) * p(x|i))))
   */

  @Override
  public void reduce(CVBKey key, Iterable<CVBTuple> values, Context context) {
    double[] docSum = null;
    for(CVBTuple tuple : values) {
      double[] docTopic = tuple.getCount(AggregationBranch.DOC_TOPIC);
      if(docSum == null) {
        docSum = new double[docTopic.length];
      }
      for(int x=0; x<docSum.length; x++) {
        docSum[x] += docTopic[x];
      }
    }
    double totalDocSum = 0;
    for(double d : docSum) {
      totalDocSum += d;
    }
    for(int x=0; x<docSum.length; x++) {
      docSum[x] /= totalDocSum;
    }
    for(CVBTuple tuple : values) {
      double[] topicTerm = tuple.getCount(AggregationBranch.TOPIC_TERM);
      double[] topicSum = tuple.getCount(AggregationBranch.TOPIC_SUM);
      double odds = 0;
      for(int x=0; x<topicTerm.length; x++) {
        odds += (topicTerm[x] / topicSum[x]) * docSum[x];
      }
      perplexity += -Math.log(odds);
    }
    // TODO: this method
  }

  @Override
  public void cleanup(Context context) throws IOException, InterruptedException {
    context.write(NullWritable.get(), new DoubleWritable(perplexity));
  }
}
