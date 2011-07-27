package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;

public class DistributedRowMatrixInputMapper
    extends Mapper<IntWritable, VectorWritable, CVBKey, CVBTuple> {
  private CVBKey outputKey = new CVBKey();
  private CVBTuple outputValue = new CVBTuple();
  @Override
  public void map(IntWritable docIdWritable, VectorWritable v, Context ctx)
      throws IOException, InterruptedException {
    outputKey.setDocId(docIdWritable.get());
    Iterator<Vector.Element> it = v.get().iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      outputKey.setTermId(e.index());
      outputKey.setBranch(AggregationBranch.TOPIC_SUM);
      outputValue.setItemCount(e.get());
      ctx.write(outputKey, outputValue);
    }
  }
}
