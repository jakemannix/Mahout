package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class CVB0DocInferenceMapper extends CachingCVB0Mapper {

  @Override
  public void map(IntWritable docId, VectorWritable doc, Context context)
      throws IOException, InterruptedException {
    Vector docTopics = new DenseVector(new double[numTopics]).assign(1d/numTopics);
    Matrix docModel = new SparseRowMatrix(numTopics, doc.get().size());
    for(int i = 0; i < maxIters; i++) {
      modelTrainer.getReadModel().trainDocTopicModel(doc.get(), docTopics, docModel);
    }
    context.write(docId, new VectorWritable(docTopics));
  }

  @Override
  protected void cleanup(Context context) {
    modelTrainer.stop();
  }
}
