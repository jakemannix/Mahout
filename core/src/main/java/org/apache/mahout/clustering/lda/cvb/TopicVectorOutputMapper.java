package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class TopicVectorOutputMapper extends Mapper<CVBKey, CVBTuple, IntWritable, VectorWritable> {

  private int numTerms;
  private final IntWritable topic = new IntWritable();
  private final VectorWritable termsVector = new VectorWritable();

  @Override
  public void map(CVBKey key, CVBTuple value, Context context)
      throws IOException, InterruptedException {
    int termId = key.getTermId();
    double[] topicTermCounts = value.getCount(AggregationBranch.TOPIC_TERM);
    double[] topicCounts = value.getCount(AggregationBranch.TOPIC_SUM);
    for(int x = 0; x < topicCounts.length; x++) {
      topic.set(x);
      RandomAccessSparseVector pTermGivenTopic = new RandomAccessSparseVector(numTerms, 1);
      pTermGivenTopic.set(termId, topicTermCounts[x] / topicCounts[x]);
      termsVector.set(pTermGivenTopic);
      context.write(topic, termsVector);
    }
  }

  @Override
  protected void setup(Context context) {
    Configuration conf = context.getConfiguration();
    numTerms = conf.getInt(CVB0Mapper.NUM_TERMS, -1);
  }
}
