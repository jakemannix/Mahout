package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class CachingCVB0PerplexityMapper extends
    Mapper<IntWritable, VectorWritable, DoubleWritable, DoubleWritable> {
  /**
   * Counters for {@link CachingCVB0PerplexityMapper}.
   */
  public static enum Counters {
    SAMPLED_DOCUMENTS;
  }

  private static Logger log = LoggerFactory.getLogger(CachingCVB0PerplexityMapper.class);
  protected ModelTrainer modelTrainer;
  protected int maxIters;
  protected int numTopics;
  protected float testFraction;
  protected Random random;
  protected Vector topicVector;
  protected final DoubleWritable outKey = new DoubleWritable();
  protected final DoubleWritable outValue = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    MemoryUtil.startMemoryLogger(5000);

    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    double eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    double alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    random = new Random(seed);
    numTopics = conf.getInt(CVB0Driver.NUM_TOPICS, -1);
    int numTerms = conf.getInt(CVB0Driver.NUM_TERMS, -1);
    int numUpdateThreads = conf.getInt(CVB0Driver.NUM_UPDATE_THREADS, 1);
    int numTrainThreads = conf.getInt(CVB0Driver.NUM_TRAIN_THREADS, 4);
    maxIters = conf.getInt(CVB0Driver.MAX_ITERATIONS_PER_DOC, 10);
    double modelWeight = conf.getFloat(CVB0Driver.MODEL_WEIGHT, 1f);
    testFraction = conf.getFloat(CVB0Driver.TEST_SET_FRACTION, 0.1f);

    log.info("Initializing read model");
    TopicModel readModel;
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);
    if(modelPaths != null && modelPaths.length > 0) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
    } else {
      log.info("No model files found");
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, new Random(seed), null,
          numTrainThreads, modelWeight);
    }

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, null, numTrainThreads, numTopics, numTerms);

    log.info("Initializing topic vector");
    topicVector = new DenseVector(new double[numTopics]);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    MemoryUtil.stopMemoryLogger();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException{
    if (1 > testFraction && random.nextFloat() >= testFraction) {
      return;
    }
    context.getCounter(Counters.SAMPLED_DOCUMENTS).increment(1);
    outKey.set(document.get().norm(1));
    outValue.set(modelTrainer.calculatePerplexity(document.get(), topicVector.assign(1.0 / numTopics), maxIters));
    context.write(outKey, outValue);
  }
}
