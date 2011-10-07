package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.Random;

public class CachingCVB0Mapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static Logger log = LoggerFactory.getLogger(CachingCVB0Mapper.class);

  protected ModelTrainer modelTrainer;
  protected int maxIters;
  protected int numTopics;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    double eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    double alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    numTopics = conf.getInt(CVB0Driver.NUM_TOPICS, -1);
    int numTerms = conf.getInt(CVB0Driver.NUM_TERMS, -1);
    int numUpdateThreads = conf.getInt(CVB0Driver.NUM_UPDATE_THREADS, 1);
    int numTrainThreads = conf.getInt(CVB0Driver.NUM_TRAIN_THREADS, 4);
    maxIters = conf.getInt(CVB0Driver.MAX_ITERATIONS_PER_DOC, 10);
    URI[] localFiles = DistributedCache.getCacheFiles(conf);
    Path[] localPaths = null;
    if(localFiles != null) {
      localPaths = new Path[localFiles.length];
      for(int i = 0; i < localFiles.length; i++) {
        localPaths[i] = new Path(localFiles[i]);
      }
    }
    TopicModel readModel;
    if(localPaths != null) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, localPaths);
    } else {
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, new Random(seed), null,
          numTrainThreads);
    }
    TopicModel writeModel = new TopicModel(numTopics, numTerms, eta, alpha, null, numUpdateThreads);
    modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
    modelTrainer.start();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException{
    /* where to get docTopics? */
    Vector topicVector = new DenseVector(new double[numTopics]).assign(1/numTopics);
    modelTrainer.train(document.get(), topicVector, true, maxIters);
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
