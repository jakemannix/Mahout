package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * ./bin/mahout distcvb0 --input /path/to/drm_data --output /output --num_topics
 */
public class CVB0Driver extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(CVB0Driver.class);

  public static final String NUM_TOPICS = "num_topics";
  public static final String NUM_TERMS = "num_terms";
  public static final String DOC_TOPIC_SMOOTHING = "doc_topic_smoothing";
  public static final String TERM_TOPIC_SMOOTHING = "term_topic_smoothing";
  public static final String DICTIONARY = "dictionary";
  public static final String DOC_TOPIC_OUTPUT = "doc_topic_output";
  public static final String MODEL_TEMP_DIR = "topic_model_temp_dir";
  public static final String ITERATION_BLOCK_SIZE = "iteration_block_size";
  public static final String RANDOM_SEED = "random_seed";
  
  @Override public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    addOption(NUM_TOPICS, "k", "Number of topics to learn", true);
    addOption(NUM_TERMS, "nt", "Vocabulary size", false);
    addOption(DOC_TOPIC_SMOOTHING, "a", "Smoothing for document/topic distribution", "0.1");
    addOption(TERM_TOPIC_SMOOTHING, "e", "Smoothing for topic/term distribution, 0.1");
    addOption(DICTIONARY, "dict", "Path to term-dictionary file", false);
    addOption(DOC_TOPIC_OUTPUT, "dt", "Output path for the training doc/topic distribution", false);
    addOption(MODEL_TEMP_DIR, "mt", "Path to intermediate model path (useful for restarting)", false);
    addOption(ITERATION_BLOCK_SIZE, "block", "Number of iterations per perplexity check", "10");
    addOption(RANDOM_SEED, "seed", "Random seed", false);

    if(parseArguments(args) == null) {
      return -1;
    }

    int numTopics = Integer.parseInt(getOption(NUM_TOPICS));
    Path inputPath = getInputPath();
    Path topicModelOutputPath = getOutputPath();
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int iterationBlockSize = Integer.parseInt(getOption(ITERATION_BLOCK_SIZE));
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    double alpha = Double.parseDouble(getOption(DOC_TOPIC_SMOOTHING));
    double eta = Double.parseDouble(getOption(TERM_TOPIC_SMOOTHING));
    Path dictionaryPath = hasOption(DICTIONARY) ? new Path(getOption(DICTIONARY)) : null;
    int numTerms = hasOption(DICTIONARY)
                 ? getNumTerms(dictionaryPath)
                 : Integer.parseInt(getOption(NUM_TERMS));
    Path docTopicOutputPath = hasOption(DOC_TOPIC_OUTPUT) ? new Path(getOption(DOC_TOPIC_OUTPUT)) : null;
    Path modelTempPath = hasOption(MODEL_TEMP_DIR)
                       ? new Path(getOption(MODEL_TEMP_DIR))
                       : getTempPath("topicModelState");
    long seed = hasOption(RANDOM_SEED)
              ? Long.parseLong(getOption(RANDOM_SEED))
              : System.nanoTime() % 10000;

    return run(getConf(), inputPath, topicModelOutputPath, numTopics, numTerms, alpha, eta,
        maxIterations, iterationBlockSize, convergenceDelta, dictionaryPath, docTopicOutputPath,
        modelTempPath, seed);
  }

  private int getNumTerms(Path dictionaryPath) {
    throw new UnsupportedOperationException("not yet!");
  }

  public int run(Configuration conf, Path inputPath, Path topicModelOutputPath, int numTopics,
      int numTerms, double alpha, double eta, int maxIterations, int iterationBlockSize,
      double convergenceDelta, Path dictionaryPath, Path docTopicOutputPath,
      Path topicModelStateTempPath, long randomSeed)
      throws ClassNotFoundException, IOException, InterruptedException {
    String infoString = "Will run Collapsed Variational Bayes (0th-derivative approximation) " +
      "learning for LDA on {}, finding {}-topics, with document/topic prior {}, topic/term " +
      "prior {}.  Maximum iterations to run will be {}, unless the change in perplexity is " +
      "less than {}.  Topic model output (p(term|topic) for each topic) will be stored {}." +
      "Random initialization seed is {}\n";
    log.info(infoString, new Object[] {inputPath, numTopics, alpha, eta, maxIterations,
        convergenceDelta, topicModelOutputPath, randomSeed});
    infoString = dictionaryPath == null
               ? "" : "Dictionary to be used located " + dictionaryPath.toString() + "\n";
    infoString += docTopicOutputPath == null
               ? "" : "p(topic|docId) will be stored " + docTopicOutputPath.toString() + "\n";
    log.info(infoString);

    double perplexity = 0;
    double previousPerplexity = Integer.MAX_VALUE;
    int iterationNumber = getCurrentIterationNumber(conf, topicModelStateTempPath);
    if(iterationNumber < 0) {
      runStage0(conf, inputPath, topicModelStateTempPath);
    }
    conf.set(CVB0Mapper.NUM_TOPICS, String.valueOf(numTopics));
    conf.set(CVB0Mapper.NUM_TERMS, String.valueOf(numTerms));
    conf.set(CVB0Mapper.ALPHA, String.valueOf(alpha));
    conf.set(CVB0Mapper.ETA, String.valueOf(eta));
    long startTime = System.currentTimeMillis();
    while(iterationNumber < maxIterations && previousPerplexity - perplexity < convergenceDelta) {
      iterationNumber++;
      Path input = stage1InputPath(topicModelStateTempPath, iterationNumber);
      Path output = stage1OutputPath(topicModelStateTempPath, iterationNumber);
      runIteration(conf, input, output, iterationNumber);
      if(iterationNumber % iterationBlockSize == 0) {
        log.warn("We would normally be spitting out perplexity here");
      }
    }
    log.info("Completed {} iterations in {} seconds",
        new Object[] {iterationNumber, (System.currentTimeMillis() - startTime)/1000} );

    return 0;
  }

  private void runStage0(Configuration conf, Path inputPath, Path topicModelStateTempPath)
      throws IOException, ClassNotFoundException, InterruptedException {
    Job job = new Job(conf, "Stage0, converting " + inputPath + "to CVB format");
    job.setMapperClass(DistributedRowMatrixInputMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    FileInputFormat.addInputPath(job, inputPath);
    FileOutputFormat.setOutputPath(job, stage1InputPath(topicModelStateTempPath, 0));
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new IOException("Unable to convert to CVB format");
    }
  }

  private Path stage1InputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "stage1-" + iterationNumber);
  }

  private Path stage1OutputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "tmp-" + iterationNumber);
  }

  private int getCurrentIterationNumber(Configuration config, Path modelTempDir)
      throws IOException {
    FileSystem fs = FileSystem.get(config);
    int iterationNumber = 0;
    Path iterationPath = stage1InputPath(modelTempDir, iterationNumber);
    while(fs.exists(iterationPath)) {
      iterationNumber++;
      iterationPath = stage1InputPath(modelTempDir, iterationNumber);
    }
    return iterationNumber - 1;
  }

  private void runIteration(Configuration conf, Path input, Path output, int iterationNumber)
      throws IOException, ClassNotFoundException, InterruptedException {
    Job job = new Job(conf, "Stage1, iteration " + iterationNumber);
    job.setMapperClass(CVB0Mapper.class);
    job.setReducerClass(CVB0Reducer.class);
    job.setCombinerClass(CVB0Reducer.class);
    job.setGroupingComparatorClass(CVB0GroupingComparator.class);
    job.setPartitionerClass(CVB0Partitioner.class);
    job.setMapOutputKeyClass(CVBKey.class);
    job.setMapOutputValueClass(CVBTuple.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to complete LDA phase 1 iteration " + iterationNumber);
    }

    job = new Job(conf, "Stage2, iteration " + iterationNumber);
    job.setMapperClass(Mapper.class);
    job.setReducerClass(CVBAggregatingReducer.class);
    job.setCombinerClass(CVBAggregatingReducer.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, output);
    FileOutputFormat.setOutputPath(job, stage1InputPath(output.getParent(), iterationNumber + 1));
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to complete aggregation (phase 2) of LDA " + iterationNumber);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CVB0Driver(), args);
  }
}
