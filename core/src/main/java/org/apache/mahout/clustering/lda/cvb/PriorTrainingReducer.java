package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;

public class PriorTrainingReducer extends MapReduceBase
    implements Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static final Logger log = LoggerFactory.getLogger(PriorTrainingReducer.class);

  public static final String DOC_TOPICS = "docTopics";
  public static final String TOPIC_TERMS = "topicTerms";
  private ModelTrainer modelTrainer;
  private int maxIters;
  private int numTopics;
  private MultipleOutputs multipleOutputs;
  private Reporter reporter;

  protected ModelTrainer getModelTrainer() {
    return modelTrainer;
  }

  protected int getMaxIters() {
    return maxIters;
  }

  protected int getNumTopics() {
    return numTopics;
  }

  @Override
  public void configure(JobConf conf) {
    try {
    log.info("Retrieving configuration");
    multipleOutputs = new MultipleOutputs(conf);
    double eta = conf.getFloat(CVB0Driver.TERM_TOPIC_SMOOTHING, Float.NaN);
    double alpha = conf.getFloat(CVB0Driver.DOC_TOPIC_SMOOTHING, Float.NaN);
    long seed = conf.getLong(CVB0Driver.RANDOM_SEED, 1234L);
    numTopics = conf.getInt(CVB0Driver.NUM_TOPICS, -1);
    int numTerms = conf.getInt(CVB0Driver.NUM_TERMS, -1);
    int numUpdateThreads = conf.getInt(CVB0Driver.NUM_UPDATE_THREADS, 1);
    int numTrainThreads = conf.getInt(CVB0Driver.NUM_TRAIN_THREADS, 4);
    maxIters = conf.getInt(CVB0Driver.MAX_ITERATIONS_PER_DOC, 10);
    double modelWeight = conf.getFloat(CVB0Driver.MODEL_WEIGHT, 1f);

    log.info("Initializing read model");
    TopicModel readModel;
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);
    if(modelPaths != null && modelPaths.length > 0) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
    } else {
      log.info("No model files found");
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, RandomUtils.getRandom(seed), null,
          numTrainThreads, modelWeight);
    }

    log.info("Initializing write model");
    TopicModel writeModel = modelWeight == 1
        ? new TopicModel(numTopics, numTerms, eta, alpha, null, numUpdateThreads)
        : readModel;

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
    modelTrainer.start();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void reduce(IntWritable docId, Iterator<VectorWritable> vectors,
      OutputCollector<IntWritable,VectorWritable> out, Reporter reporter)
      throws IOException {
    if(this.reporter == null) {
      this.reporter = reporter;
    }
    Vector topicVector = new DenseVector(new double[numTopics]).assign(1.0/numTopics);
    Vector document = null;
    while(vectors.hasNext()) {
      VectorWritable v = vectors.next();
      if(v.get().isDense()) {
        topicVector = v.get();
      } else {
        document = v.get();
      }
    }
    if(document != null) {
      modelTrainer.trainSync(document, topicVector, true, 1);
      multipleOutputs.getCollector(DOC_TOPICS, reporter)
                     .collect(docId, new VectorWritable(topicVector));
    }
  }

  @Override
  public void close() throws IOException {
    log.info("Stopping model trainer");
    modelTrainer.stop();

    log.info("Writing model");
    TopicModel model = modelTrainer.getReadModel();
    for(MatrixSlice topic : model) {
      multipleOutputs.getCollector(TOPIC_TERMS, reporter)
                     .collect(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
    multipleOutputs.close();
  }
}
