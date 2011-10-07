package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.Random;

public class CachingCVB0Mapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static Logger log = LoggerFactory.getLogger(CachingCVB0Mapper.class);

  private ModelTrainer modelTrainer;
  private int maxIters;
  private int numTopics;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    URI[] localFiles = DistributedCache.getCacheFiles(conf);
    Path[] localPaths = new Path[localFiles.length];
    for(int i = 0; i < localFiles.length; i++) {
      localPaths[i] = new Path(localFiles[i]);
    }
    double eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    double alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    int numThreads = 4; // TODO: configure!
    int numTrainThreads = 10; // TODO: configure!
    maxIters = 1; // TODO: configure!
    TopicModel readModel = new TopicModel(conf, eta, alpha, null, numThreads, localPaths);
    numTopics = readModel.topicTermCounts().numRows();
    int numTerms = readModel.topicTermCounts().numCols();
    TopicModel writeModel = new TopicModel(numTopics, numTerms, eta, alpha, new Random(seed), null,
        numThreads);
    modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context) {
    modelTrainer.train(document.get(),
        new DenseVector(new double[numTopics]).assign(1/numTopics) /* where to get docTopics? */,
        true, maxIters);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    modelTrainer.stop();
    TopicModel model = modelTrainer.getReadModel();
    for(MatrixSlice topic : model) {
      context.write(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
  }
}
