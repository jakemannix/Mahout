package org.apache.mahout.clustering.lda.cvb;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.apache.mahout.vectorizer.common.PartialVectorMergeReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Usage: <code>./bin/mahout cvb <i>options</i></code>
 * <p>
 * Valid options include:
 * <dl>
 * <dt>{@code --input path}</td>
 * <dd>Input path for {@code SequenceFile<IntWritable, VectorWritable>} document vectors. See
 * {@link SparseVectorsFromSequenceFiles} for details on how to generate this input format.</dd>
 * <dt>{@code --dictionary path}</dt>
 * <dd>Path to dictionary file(s) generated during construction of input document vectors (glob
 * expression supported). If set, this data is scanned to determine an appropriate value for option
 * {@code --num_terms}.</dd>
 * <dt>{@code --output path}</dt>
 * <dd>Output path for topic-term distributions.</dd>
 * <dt>{@code --doc_topic_output path}</dt>
 * <dd>Output path for doc-topic distributions.</dd>
 * <dt>{@code --num_topics k}</dt>
 * <dd>Number of latent topics.</dd>
 * <dt>{@code --num_terms nt}</dt>
 * <dd>Number of unique features defined by input document vectors. If option {@code --dictionary}
 * is defined, this option is ignored.</dd>
 * <dt>{@code --topic_model_temp_dir path}</dt>
 * <dd>Path in which to store model state after each iteration.</dd>
 * <dt>{@code --maxIter i}</dt>
 * <dd>Maximum number of iterations to perform. If this value is less than or equal to the number of
 * iteration states found beneath the path specified by option {@code --topic_model_temp_dir}, no
 * further iterations are performed. Instead, output topic-term and doc-topic distributions are
 * generated using data from the specified iteration.</dd>
 * <dt>{@code --doc_topic_smoothing a}</dt>
 * <dd>Smoothing for doc-topic distribution. Defaults to {@code 0.1}.</dd>
 * <dt>{@code --term_topic_smoothing e}</dt>
 * <dd>Smoothing for topic-term distribution. Defaults to {@code 0.1}.</dd>
 * <dt>{@code --random_seed seed}</dt>
 * <dd>Integer seed for random number generation.</dd>
 * <dt>{@code --test_set_percentage p}</dt>
 * <dd>Fraction of data to hold out for testing. Defaults to {@code 0.0}.</dd>
 * <dt>{@code --iteration_block_size block}</dt>
 * <dd>Number of iterations between perplexity checks. Defaults to {@code 10}. This option is
 * ignored unless option {@code --test_set_percentage} is greater than zero.</dd>
 * </dl>
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
  public static final String TEST_SET_PERCENTAGE = "test_set_percentage";

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
    addOption(DICTIONARY, "dict", "Path to term-dictionary file(s) (glob expression supported)", false);
    addOption(DOC_TOPIC_OUTPUT, "dt", "Output path for the training doc/topic distribution", false);
    addOption(MODEL_TEMP_DIR, "mt", "Path to intermediate model path (useful for restarting)", false);
    addOption(ITERATION_BLOCK_SIZE, "block", "Number of iterations per perplexity check", "10");
    addOption(RANDOM_SEED, "seed", "Random seed", false);
    addOption(TEST_SET_PERCENTAGE, "tp", "% of data to hold out for testing", false);

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
                 ? getNumTerms(getConf(), dictionaryPath)
                 : Integer.parseInt(getOption(NUM_TERMS));
    Path docTopicOutputPath = hasOption(DOC_TOPIC_OUTPUT) ? new Path(getOption(DOC_TOPIC_OUTPUT)) : null;
    Path modelTempPath = hasOption(MODEL_TEMP_DIR)
                       ? new Path(getOption(MODEL_TEMP_DIR))
                       : getTempPath("topicModelState");
    long seed = hasOption(RANDOM_SEED)
              ? Long.parseLong(getOption(RANDOM_SEED))
              : System.nanoTime() % 10000;
    float testFraction = hasOption(TEST_SET_PERCENTAGE)
                       ? Float.parseFloat(getOption(TEST_SET_PERCENTAGE))
                       : 0f;

    return run(getConf(), inputPath, topicModelOutputPath, numTopics, numTerms, alpha, eta,
        maxIterations, iterationBlockSize, convergenceDelta, dictionaryPath, docTopicOutputPath,
        modelTempPath, seed, testFraction);
  }

  private int getNumTerms(Configuration conf, Path dictionaryPath) throws IOException {
    FileSystem fs = dictionaryPath.getFileSystem(conf);
    Text key = new Text();
    IntWritable value = new IntWritable();
    int maxTermId = -1;
    for (FileStatus stat : fs.globStatus(dictionaryPath)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, stat.getPath(), conf);
      while (reader.next(key, value)) {
        maxTermId = Math.max(maxTermId, value.get());
      }
    }
    return maxTermId + 1;
  }

  public int run(Configuration conf, Path inputPath, Path topicModelOutputPath, int numTopics,
      int numTerms, double alpha, double eta, int maxIterations, int iterationBlockSize,
      double convergenceDelta, Path dictionaryPath, Path docTopicOutputPath,
      Path topicModelStateTempPath, long randomSeed, float testFraction)
      throws ClassNotFoundException, IOException, InterruptedException {
    String infoString = "Will run Collapsed Variational Bayes (0th-derivative approximation) " +
      "learning for LDA on {} (numTerms: {}), finding {}-topics, with document/topic prior {}, " +
      "topic/term prior {}.  Maximum iterations to run will be {}, unless the change in " +
      "perplexity is less than {}.  Topic model output (p(term|topic) for each topic) will be " +
      "stored {}.  Random initialization seed is {}, holding out {} of the data for perplexity " +
      "check\n";
    log.info(infoString, new Object[] {inputPath, numTerms, numTopics, alpha, eta, maxIterations,
        convergenceDelta, topicModelOutputPath, randomSeed, testFraction});
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
      iterationNumber = 0;
    } else if (iterationNumber > maxIterations) {
      iterationNumber = maxIterations;
    }
    conf.set(CVB0Mapper.NUM_TOPICS, String.valueOf(numTopics));
    conf.set(CVB0Mapper.NUM_TERMS, String.valueOf(numTerms));
    conf.set(CVB0Mapper.ALPHA, String.valueOf(alpha));
    conf.set(CVB0Mapper.ETA, String.valueOf(eta));
    conf.set(CVB0Mapper.RANDOM_SEED, String.valueOf(randomSeed));
    conf.set(CVB0Mapper.TEST_SET_PCT, String.valueOf(testFraction));
    long startTime = System.currentTimeMillis();
    while(iterationNumber < maxIterations && previousPerplexity - perplexity > convergenceDelta) {
      iterationNumber++;
      log.info("About to run iteration " + iterationNumber);
      Path stage1input = stage1InputPath(topicModelStateTempPath, iterationNumber - 1);
      Path stage1output = stage1OutputPath(topicModelStateTempPath, iterationNumber - 1);
      runIteration(conf, stage1input, stage1output, iterationNumber);
      if(testFraction > 0 && iterationNumber % iterationBlockSize == 0) {
        previousPerplexity = perplexity;
        perplexity = calculatePerplexity(conf, stage1output);
        log.info("current perplexity = " + perplexity);
      }
    }
    log.info("Completed {} iterations in {} seconds",
        new Object[] {iterationNumber, (System.currentTimeMillis() - startTime)/1000} );
    Path finalIterationData = stage1InputPath(topicModelStateTempPath, iterationNumber);
    Job topicWritingJob = writeTopicModelVectors(finalIterationData, topicModelOutputPath, numTerms);
    Job docInferenceJob = docTopicOutputPath != null
        ? writeDocTopicInference(finalIterationData, docTopicOutputPath)
        : null;
    try {
      if(!topicWritingJob.waitForCompletion(true)) {
        return -1;
      }
      if(docInferenceJob != null && !docInferenceJob.waitForCompletion(true)) {
        return -1;
      }
    } finally {
      // get rid of temp paths
      Path[] tmpPaths = FileInputFormat.getInputPaths(topicWritingJob);
      if (tmpPaths != null && tmpPaths.length > 0) {
        try {
          HadoopUtil.delete(conf, tmpPaths);
        } catch (Exception e) {
          log.error("Failed to delete temp paths '" + Arrays.toString(tmpPaths) + "'");
        }
      }
    }
    return 0;
  }

  private double calculatePerplexity(Configuration conf, Path stage1output) throws IOException,
      ClassNotFoundException, InterruptedException {
    String jobName = "Calculating perplexity for " + stage1output;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(PerplexityCheckingMapper.class);
    //job.setReducerClass(PerplexityCheckingReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    FileInputFormat.addInputPath(job, stage1output);
    Path outputPath = new Path(stage1output.getParent(), "perplexity");
    FileOutputFormat.setOutputPath(job, outputPath);
    HadoopUtil.delete(conf, outputPath);
    job.setJarByClass(PerplexityCheckingMapper.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to calculate perplexity for: " + stage1output);
    }
    double perplexity = 0;
    FileSystem fs = FileSystem.get(conf);
    FileStatus[] statuses = fs.listStatus(outputPath, PathFilters.partFilter());
    for(FileStatus status : statuses) {
      log.info("reading perplexity from: " + status.getPath());
      int i = 0;
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
      DoubleWritable d = new DoubleWritable();
      while(reader.next(NullWritable.get(), d)) {
        perplexity += d.get();
        i++;
      }
      log.info("read " + i + " perplexity values");
    }
    return perplexity;
  }

  private Job writeTopicModelVectors(Path input, Path output, int numTerms)
      throws ClassNotFoundException, IOException, InterruptedException {
    String jobName = "Writing final topic model (as vectors, dimension: " + numTerms + ") from "
                     + input + " to " + output;
    log.info("About to run: " + jobName);
    Configuration conf = new Configuration(getConf());
    Job job = new Job(conf, jobName);
    job.setMapperClass(TermDedupingMapper.class);
    job.setCombinerClass(TopicTermOutputReducer.class);
    job.setReducerClass(TopicTermOutputReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    FileInputFormat.addInputPath(job, input);
    Path intermediatePath = new Path(input.getParent(), "topicModelTemp-" + System.nanoTime());
    FileOutputFormat.setOutputPath(job, intermediatePath);
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Could not complete: " + jobName);
    }
    conf = new Configuration(getConf());
    conf.set(PartialVectorMerger.DIMENSION, String.valueOf(numTerms));
    conf.set(CVB0Mapper.NUM_TERMS, String.valueOf(numTerms));
    job = new Job(conf, jobName);
    job.setMapperClass(TopicVectorOutputMapper.class);
    job.setCombinerClass(PartialVectorMergeReducer.class);
    job.setReducerClass(PartialVectorMergeReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.addInputPath(job, intermediatePath);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    job.submit();
    return job;
  }

  private Job writeDocTopicInference(Path input, Path output)
      throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = "Writing final document/topic inference from " + input + " to " + output;
    log.info("About to run: " + jobName);
    Job job = new Job(getConf(), jobName);
    job.setMapperClass(DocTopicOutputMapper.class);
    job.setCombinerClass(DocTopicOutputReducer.class);
    job.setReducerClass(DocTopicOutputReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    job.submit();
    return job;
  }

  public void runStage0(Configuration conf, Path inputPath, Path topicModelStateTempPath)
      throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = "Stage 0, converting " + inputPath + " to CVB format";
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
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

  public static Path stage1InputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "stage1-" + iterationNumber);
  }

  public static Path stage1OutputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "tmp-" + iterationNumber);
  }

  private int getCurrentIterationNumber(Configuration config, Path modelTempDir)
      throws IOException {
    FileSystem fs = FileSystem.get(config);
    int iterationNumber = 0;
    Path iterationPath = stage1InputPath(modelTempDir, iterationNumber);
    while(fs.exists(iterationPath)) {
      log.info("found previous state: " + iterationPath);
      iterationNumber++;
      iterationPath = stage1InputPath(modelTempDir, iterationNumber);
    }
    return iterationNumber - 1;
  }

  public void runIterationStage1(Configuration conf, Path stage1input, Path stage1output,
      int iterationNumber) throws IOException, ClassNotFoundException, InterruptedException {
    conf.setInt(CVB0Mapper.ITERATION_NUM, iterationNumber);
    String jobName = "Stage 1, iteration " + iterationNumber + ", input path: " + stage1input;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(CVB0Mapper.class);
    job.setReducerClass(CVB0Reducer.class);
    job.setCombinerClass(CVB0Combiner.class);
    job.setGroupingComparatorClass(CVB0GroupingComparator.class);
    job.setSortComparatorClass(CVBSortingComparator.class);
    job.setPartitionerClass(CVB0Partitioner.class);
    job.setMapOutputKeyClass(CVBKey.class);
    job.setMapOutputValueClass(CVBTuple.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, stage1input);
    FileOutputFormat.setOutputPath(job, stage1output);
    HadoopUtil.delete(conf, stage1output);
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to complete LDA phase 1 iteration " + iterationNumber);
    }
  }

  public void runIterationStage2(Configuration conf, Path stage1input, Path stage1output,
      int iterationNumber) throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = "Stage 2, iteration " + iterationNumber + ", input path: " + stage1output;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(Mapper.class);
    job.setReducerClass(CVBAggregatingReducer.class);
    job.setSortComparatorClass(CVBSortingComparator.class);
    //job.setCombinerClass(CVBAggregatingReducer.class);
    job.setOutputKeyClass(CVBKey.class);
    job.setOutputValueClass(CVBTuple.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, stage1output);
    FileOutputFormat.setOutputPath(job, stage1InputPath(stage1output.getParent(), iterationNumber));
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to complete aggregation (phase 2) of LDA " + iterationNumber);
    }
  }

  public void runIteration(Configuration conf, Path stage1input, Path stage1output, int iterationNumber)
      throws IOException, ClassNotFoundException, InterruptedException {
    runIterationStage1(conf, stage1input, stage1output, iterationNumber);
    runIterationStage2(conf, stage1input, stage1output, iterationNumber);
  }

  private static class TopicOutputReducer extends UniquingReducer<IntPairWritable, DoubleWritable> {}

  private static class TopicTermOutputReducer extends UniquingReducer<CVBKey, CVBTuple> {}

  private static class DocTopicOutputReducer extends UniquingReducer<IntWritable, VectorWritable> {}

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CVB0Driver(), args);
  }
}
