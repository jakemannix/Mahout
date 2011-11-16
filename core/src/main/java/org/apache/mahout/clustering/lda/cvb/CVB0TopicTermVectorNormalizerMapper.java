package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

/**
 * Performs L1 normalization of input vectors.
 */
public class CVB0TopicTermVectorNormalizerMapper extends
    Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

  @Override
  protected void map(IntWritable key, VectorWritable value, Context context) throws IOException,
      InterruptedException {
    value.get().assign(Functions.div(value.get().norm(1.0)));
    context.write(key, value);
  }
}
