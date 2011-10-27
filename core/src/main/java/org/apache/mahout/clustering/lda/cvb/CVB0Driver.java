package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.entropy.DoubleSumReducer;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.List;

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
  public static final String NUM_TRAIN_THREADS = "num_train_threads";
  public static final String NUM_UPDATE_THREADS = "num_update_threads";
  public static final String MAX_ITERATIONS_PER_DOC = "max_doc_topic_iters";
  public static final String MODEL_WEIGHT = "prev_iter_mult";

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
    addOption(DICTIONARY, "dict", "Path to term-dictionary file(s) (glob expression supported)",
        false);
    addOption(DOC_TOPIC_OUTPUT, "dt", "Output path for the training doc/topic distribution",
        false);
    addOption(MODEL_TEMP_DIR, "mt", "Path to intermediate model path (useful for restarting)",
        false);
    addOption(ITERATION_BLOCK_SIZE, "block", "Number of iterations per perplexity check", "10");
    addOption(RANDOM_SEED, "seed", "Random seed", false);
    addOption(TEST_SET_PERCENTAGE, "tp", "% of data to hold out for testing", false);
    addOption(NUM_TRAIN_THREADS, "ntt", "number of threads per mapper to train with", "4");
    addOption(NUM_UPDATE_THREADS, "nut", "number of threads per mapper to update the model with",
        "1");
    addOption(MAX_ITERATIONS_PER_DOC, "mipd",
        "max number of iterations per doc for p(topic|doc) learning", "100");

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
    int numTrainThreads = hasOption(NUM_TRAIN_THREADS) ?
        Integer.parseInt(getOption(NUM_TRAIN_THREADS)) : 4;
    int numUpdateThreads = hasOption(NUM_UPDATE_THREADS) ?
        Integer.parseInt(getOption(NUM_UPDATE_THREADS)) : 1;
    int maxItersPerDoc = hasOption(MAX_ITERATIONS_PER_DOC) ?
        Integer.parseInt(getOption(MAX_ITERATIONS_PER_DOC)) : 10;
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
        modelTempPath, seed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc);
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
      Path topicModelStateTempPath, long randomSeed, float testFraction, int numTrainThreads,
      int numUpdateThreads, int maxItersPerDoc)
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

    List<Double> perplexities = Lists.newArrayList();
    int iterationNumber = getCurrentIterationNumber(conf, topicModelStateTempPath);
    if (iterationNumber > maxIterations) {
      iterationNumber = maxIterations;
    }
    for(int i = 0; i < iterationNumber; i++) {
      double perplexity = readPerplexity(topicModelStateTempPath, i);
      if(!Double.isNaN(perplexity)) {
        perplexities.add(perplexity);
      }
    }
    conf.set(NUM_TOPICS, String.valueOf(numTopics));
    conf.set(NUM_TERMS, String.valueOf(numTerms));
    conf.set(DOC_TOPIC_SMOOTHING, String.valueOf(alpha));
    conf.set(TERM_TOPIC_SMOOTHING, String.valueOf(eta));
    conf.set(RANDOM_SEED, String.valueOf(randomSeed));
    conf.set(NUM_TRAIN_THREADS, String.valueOf(numTrainThreads));
    conf.set(NUM_UPDATE_THREADS, String.valueOf(numUpdateThreads));
    conf.set(MAX_ITERATIONS_PER_DOC, String.valueOf(maxItersPerDoc));
    conf.set(MODEL_WEIGHT, String.valueOf(1)); // TODO:
    double modelWeight = -1;
    long startTime = System.currentTimeMillis();
    while(iterationNumber < maxIterations && (perplexities.size() > 1 ||
          rateOfChange(perplexities) > convergenceDelta)) {
      iterationNumber++;
      log.info("About to run iteration {} of {}", iterationNumber, maxIterations);
      Path modelInputPath = modelPath(topicModelStateTempPath, iterationNumber - 1);
      Path modelOutputPath = modelPath(topicModelStateTempPath, iterationNumber);
      runIteration(conf, inputPath, modelInputPath, modelOutputPath, iterationNumber, maxIterations);
      if(modelWeight < 0) {
        modelWeight = calculateModelWeight(conf, modelOutputPath);
      }
      if(testFraction > 0 && iterationNumber % iterationBlockSize == 0) {
        perplexities.add(calculatePerplexity(conf, modelOutputPath, iterationNumber) / modelWeight);
        log.info("Current perplexity = " + perplexities.get(perplexities.size() - 1));
        log.info("p_" + iterationNumber + " - p_" + (iterationNumber-iterationBlockSize) +
                 " / p_0 = " + rateOfChange(perplexities) + ".  target: " + convergenceDelta);
      }
    }
    log.info("Completed {} iterations in {} seconds", iterationNumber,
        (System.currentTimeMillis() - startTime)/1000);
    log.info("Perplexities: (" + Joiner.on(", ").join(perplexities) + ")");
    Path finalIterationData = modelPath(topicModelStateTempPath, iterationNumber);
    Job docInferenceJob = docTopicOutputPath != null
        ? writeDocTopicInference(conf, inputPath, finalIterationData, docTopicOutputPath)
        : null;
    if(docInferenceJob != null && !docInferenceJob.waitForCompletion(true)) {
      return -1;
    }
    return 0;
  }

  private double rateOfChange(List<Double> perplexities) {
    int sz = perplexities.size();
    if(sz < 2) {
      return 1;
    }
    return Math.abs(perplexities.get(sz - 1) - perplexities.get(sz - 2)) / perplexities.get(0);
  }

  private double calculatePerplexity(Configuration conf, Path stage1output, int iteration)
      throws IOException,
      ClassNotFoundException, InterruptedException {
    String jobName = "Calculating perplexity for " + stage1output;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CachingCVB0PerplexityMapper.class);
    job.setMapperClass(CachingCVB0PerplexityMapper.class);
    job.setReducerClass(DoubleSumReducer.class);
    job.setCombinerClass(DoubleSumReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    FileInputFormat.addInputPath(job, stage1output);
    Path outputPath = perplexityPath(stage1output.getParent(), iteration);
    HadoopUtil.delete(conf, outputPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    HadoopUtil.delete(conf, outputPath);
    job.setJarByClass(PerplexityCheckingMapper.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to calculate perplexity for: " + stage1output);
    }
    return readPerplexity(stage1output.getParent(), iteration);
  }

  private double readPerplexity(Path topicModelStateTemp, int iteration) throws IOException {
    Path perplexityPath = perplexityPath(topicModelStateTemp, iteration);
    double perplexity = 0;
    FileSystem fs = FileSystem.get(getConf());
    if(!fs.exists(perplexityPath)) {
      log.warn("Perplexity path " + perplexityPath + " does not exist, returning NaN");
      return Double.NaN;
    }
    FileStatus[] statuses = fs.listStatus(perplexityPath, PathFilters.partFilter());
    for(FileStatus status : statuses) {
      log.info("Reading perplexity from: " + status.getPath());
      int i = 0;
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), getConf());
      DoubleWritable d = new DoubleWritable();
      while(reader.next(NullWritable.get(), d)) {
        perplexity += d.get();
        i++;
      }
      log.info("Read " + i + " perplexity values");
    }
    return perplexity;

  }

  private Job writeDocTopicInference(Configuration conf, Path corpus, Path modelInput, Path output)
      throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = String.format("Writing final document/topic inference from %s to %s", corpus,
        output);
    log.info("About to run: " + jobName);
    Job job = new Job(getConf(), jobName);
    job.setMapperClass(CVB0DocInferenceMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileSystem fs = FileSystem.get(conf);
    if(modelInput != null && fs.exists(modelInput)) {
      FileStatus[] statuses = FileSystem.get(conf).listStatus(modelInput, PathFilters.partFilter());
      URI[] modelUris = new URI[statuses.length];
      for(int i = 0; i < statuses.length; i++) {
        modelUris[i] = statuses[i].getPath().toUri();
      }
      DistributedCache.setCacheFiles(modelUris, conf);
    }
    FileInputFormat.addInputPath(job, corpus);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    job.submit();
    return job;
  }

  public static Path modelPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "model-" + iterationNumber);
  }

  public static Path stage1OutputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "tmp-" + iterationNumber);
  }

  public static Path perplexityPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "perplexity-" + iterationNumber);
  }

  private int getCurrentIterationNumber(Configuration config, Path modelTempDir)
      throws IOException {
    FileSystem fs = FileSystem.get(config);
    int iterationNumber = 0;
    Path iterationPath = modelPath(modelTempDir, iterationNumber);
    while(fs.exists(iterationPath)) {
      log.info("found previous state: " + iterationPath);
      iterationNumber++;
      iterationPath = modelPath(modelTempDir, iterationNumber);
    }
    return iterationNumber - 1;
  }

  public void runIteration(Configuration conf, Path corpusInput, Path modelInput, Path modelOutput,
      int iterationNumber, int maxIterations) throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = String.format("Iteration %d of %d, input path: %s",
        iterationNumber, maxIterations, modelInput);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(CachingCVB0Mapper.class);
    job.setReducerClass(VectorSumReducer.class);
    job.setCombinerClass(VectorSumReducer.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, corpusInput);
    FileOutputFormat.setOutputPath(job, modelOutput);
    FileSystem fs = FileSystem.get(conf);
    if(modelInput != null && fs.exists(modelInput)) {
      FileStatus[] statuses = FileSystem.get(conf).listStatus(modelInput, PathFilters.partFilter());
      URI[] modelUris = new URI[statuses.length];
      for(int i = 0; i < statuses.length; i++) {
        modelUris[i] = statuses[i].getPath().toUri();
      }
      DistributedCache.setCacheFiles(modelUris, conf);
    }
    HadoopUtil.delete(conf, modelOutput);
    job.setJarByClass(CVB0Driver.class);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException(String.format("Failed to complete iteration %d stage 1",
          iterationNumber));
    }
  }

  private double calculateModelWeight(Configuration conf, Path modelBasePath) throws IOException {
    Path[] modelPaths = Collections2.transform(Lists.<FileStatus>newArrayList(
        FileSystem.get(conf).listStatus(modelBasePath, PathFilters.partFilter())),
        new Function<FileStatus, Path>() {
          @Override public Path apply(FileStatus fileStatus) {
            return fileStatus.getPath();
          }
    }).toArray(new Path[0]);
    
    double weight = 0;
    for(Path modelPath : modelPaths) {
      for(Pair<IntWritable, VectorWritable> row :
          new SequenceFileIterable<IntWritable, VectorWritable>(modelPath, true, conf)) {
        weight += row.getSecond().get().norm(1);
      }
    }
    return weight;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CVB0Driver(), args);
  }
}
