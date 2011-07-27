package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class DocTopicOutputMapper extends Mapper<CVBKey, CVBTuple, IntWritable, VectorWritable> {

  private final IntWritable docId = new IntWritable();
  private final VectorWritable docTopicsVector = new VectorWritable();

  @Override
  public void map(CVBKey key, CVBTuple tuple, Context context)
      throws IOException, InterruptedException {
    docId.set(key.getDocId());
    docTopicsVector.set(new DenseVector(tuple.getCount(AggregationBranch.DOC_TOPIC)).normalize(1));
    context.write(docId, docTopicsVector);
  }
}
