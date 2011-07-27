package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.IntPairWritable;

import java.io.IOException;

public class TopicOutputMapper extends Mapper<CVBKey, CVBTuple, IntPairWritable, DoubleWritable> {

  private final IntPairWritable topicAndTermOutput = new IntPairWritable();
  private final DoubleWritable pTermGivenTopic = new DoubleWritable();

  @Override
  public void map(CVBKey key, CVBTuple value, Context context)
      throws IOException, InterruptedException {
    topicAndTermOutput.setSecond(key.getTermId());
    double[] topicTermCounts = value.getCount(AggregationBranch.TOPIC_TERM);
    double[] topicCounts = value.getCount(AggregationBranch.TOPIC_SUM);
    for(int x = 0; x < topicCounts.length; x++) {
      pTermGivenTopic.set(topicTermCounts[x] / topicCounts[x]);
      topicAndTermOutput.setFirst(x);
      context.write(topicAndTermOutput, pTermGivenTopic);
    }
  }

}
