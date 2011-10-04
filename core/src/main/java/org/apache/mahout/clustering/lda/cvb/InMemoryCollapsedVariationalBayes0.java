package org.apache.mahout.clustering.lda.cvb;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Resources;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DistributedRowMatrixWriter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class InMemoryCollapsedVariationalBayes0 extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(InMemoryCollapsedVariationalBayes0.class);

  private int numTopics;
  private int numTerms;
  private int numDocuments;
  private double alpha;
  private double eta;
  private int minDfCt;
  private double maxDfPct;
  private boolean verbose = false;

  private Map<String, Integer> termIdMap;
  private String[] terms;  // of length numTerms;

  private Matrix corpusWeights; // length numDocs;
  private double totalCorpusWeight;

  private Matrix docTopicCounts;

  private TopicModel topicModel;
  private TopicModel updatedModel;

  private int numTrainingThreads;
  private int numUpdatingThreads;

  private ModelTrainer modelTrainer;

  private InMemoryCollapsedVariationalBayes0() {
    // only for main usage
  }

  public InMemoryCollapsedVariationalBayes0(Map<Integer, Map<String, Integer>> corpus,
      int numTopics, double alpha, double eta, int minDfCt, double maxDfPct) {
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    this.minDfCt = minDfCt;
    this.maxDfPct = maxDfPct;
    initializeCorpusWeights(corpus);
    initializeModel();
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }

  public InMemoryCollapsedVariationalBayes0(Matrix corpus, String[] terms, int numTopics,
      double alpha, double eta) {
    this(corpus, terms, numTopics, alpha, eta, 1, 1);
  }

  public InMemoryCollapsedVariationalBayes0(Matrix corpus, String[] terms, int numTopics,
      double alpha, double eta, int numTrainingThreads, int numUpdatingThreads) {
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    this.minDfCt = 0;
    this.maxDfPct = 1.0f;
    corpusWeights = corpus;
    numDocuments = corpus.numRows();
    this.terms = terms;
    numTerms = terms.length;
    termIdMap = Maps.newHashMap();
    for(int t=0; t<terms.length; t++) {
      termIdMap.put(terms[t], t);
    }
    this.numTrainingThreads = numTrainingThreads;
    this.numUpdatingThreads = numUpdatingThreads;
    postInitCorpus();
    initializeModel();
  }

  private void postInitCorpus() {
    totalCorpusWeight = 0;
    int numNonZero = 0;
    for(int i=0; i<numDocuments; i++) {
      Vector v = corpusWeights.getRow(i);
      double norm;
      if(v != null && (norm = v.norm(1)) != 0) {
        numNonZero += v.getNumNondefaultElements();
        totalCorpusWeight += norm;
      }
    }
    String s = "Initializing corpus with %d docs, %d terms, %d nonzero entries, total termWeight %f";
    log.info(String.format(s, numDocuments, numTerms, numNonZero, totalCorpusWeight));
  }

  private void initializeCorpusWeights(Map<Integer, Map<String, Integer>> corpus) {
    numDocuments = corpus.size();
    Map<String, Integer> termCounts = termCount(corpus);
    terms = termCounts.keySet().toArray(new String[termCounts.size()]);
    numTerms = terms.length;
    double[] termWeights = new double[terms.length];
    for(int t=0; t<terms.length; t++) {
      // Calculate the idf
      termWeights[t] = Math.log(1 + (1 + numDocuments) / (1 + termCounts.get(terms[t])));
    }
    termIdMap = Maps.newHashMap();
    for(int t=0; t<terms.length; t++) {
      termIdMap.put(terms[t], t);
    }
    corpusWeights = new SparseRowMatrix(new int[]{numDocuments, numTerms}, true);
    for(int i=0; i<numDocuments; i++) {
      Map<String, Integer> document = corpus.get(i);
      Vector docVector = new RandomAccessSparseVector(numTerms, document.size());
      for(Map.Entry<String, Integer> e : document.entrySet()) {
        if(termIdMap.containsKey(e.getKey())) {
          int termId = termIdMap.get(e.getKey());
          docVector.set(termId, e.getValue() * termWeights[termId]);
        }
      }
      double norm = docVector.getNumNondefaultElements();
      if(norm > 0) {
        corpusWeights.assignRow(i, docVector);
      } else {
        log.warn("Empty document vector at docId( " + i + ")");
      }
    }
    postInitCorpus();
  }

  private Map<String, Integer> termCount(Map<Integer, Map<String, Integer>> corpus) {
    Map<String, Integer> termCounts = Maps.newHashMap();
    for(int docId : corpus.keySet()) {
      for(Map.Entry<String, Integer> e : corpus.get(docId).entrySet()) {
        String term = e.getKey();
        if(!termCounts.containsKey(term)) {
          termCounts.put(term, 0);
        }
        termCounts.put(term, termCounts.get(term) + 1); // only count document frequencies here
      }
    }
    Iterator<Map.Entry<String,Integer>> it = termCounts.entrySet().iterator();
    Map.Entry<String, Integer> e = null;
    while(it.hasNext() && (e = it.next()) != null) {
      // trim out terms which are too frequent (dfPct > maxDfPct) or too rare (dfCt < minDfCt)
      float df = (float)e.getValue();
      if(df/numDocuments > maxDfPct) {
        log.info(e.getKey() + " occurs " + df + " times, removing");
        it.remove();
      } else if(df < minDfCt) {
        it.remove();
      }
    }
    return termCounts;
  }

  private void initializeModel() {
    topicModel = new TopicModel(numTopics, numTerms, eta, alpha, new Random(1234), terms,
        numUpdatingThreads);
    topicModel.setConf(getConf());
    updatedModel = new TopicModel(numTopics, numTerms, eta, alpha, null, terms, numUpdatingThreads);
    updatedModel.setConf(getConf());
    docTopicCounts = new DenseMatrix(numDocuments, numTopics);
    docTopicCounts.assign(1/numTopics);
    modelTrainer = new ModelTrainer(topicModel, updatedModel, numTrainingThreads, numTopics, numTerms);
  }

  private void inferDocuments(double convergence, int maxIter, boolean recalculate) {
    for(int docId = 0; docId < corpusWeights.numRows() ; docId++) {
      Vector inferredDocument = topicModel.infer(corpusWeights.getRow(docId),
          docTopicCounts.getRow(docId));
      // do what now?
    }
  }

  public void trainDocuments() {
    long start = System.nanoTime();
    modelTrainer.train(corpusWeights, docTopicCounts);
    logTime("train documents", System.nanoTime() - start);
  }

  private double error(int docId) {
    Vector docTermCounts = corpusWeights.getRow(docId);
    if(docTermCounts == null) {
      return 0;
    } else {
      Vector expectedDocTermCounts =
          topicModel.infer(corpusWeights.getRow(docId), docTopicCounts.getRow(docId));
      double expectedNorm = expectedDocTermCounts.norm(1);
      return expectedDocTermCounts.times(docTermCounts.norm(1)/expectedNorm)
          .minus(docTermCounts).norm(1);
    }
  }

  private double error() {
    long time = System.nanoTime();
    double error = 0;
    for(int docId = 0; docId < numDocuments; docId++) {
      error += error(docId);
    }
    logTime("error calculation", System.nanoTime() - time);
    return error / totalCorpusWeight;
  }

  public double iterateUntilConvergence(double minFractionalErrorChange, int maxIterations, int minIter) {
    double fractionalChange = Double.MAX_VALUE;
    int iter = 0;
    double oldError = 0;
    while(iter < minIter) {
      if(verbose) {
        log.info(modelTrainer.getReadModel().toString());
      }
      trainDocuments();
      log.info("iteration " + iter + " complete");
      oldError = 0; //error();
      log.info(oldError + " = error");
      iter++;
    }
    double newError = 0;
    while(iter < maxIterations && fractionalChange > minFractionalErrorChange) {
      trainDocuments();
      log.info("iteration " + iter + " complete");
      newError = error();
      iter++;
      fractionalChange = Math.abs(newError - oldError) / oldError;
      log.info(fractionalChange + " = fractionalChange");
      oldError = newError;
    }
    if(iter < maxIterations) {
      log.info(String.format("Converged! fractional error change: %f, error %f",
          fractionalChange, newError));
    } else {
      log.info(String.format("Reached max iteration count (%d), fractional error change: %f, error: %f",
          maxIterations, fractionalChange, newError));
    }
    return newError;
  }

  public void writeModel(Path outputPath) throws IOException {
    modelTrainer.persist(outputPath);
  }

  private static final void logTime(String label, long nanos) {
    log.info(label + " time: " + (double)(nanos)/1e6 + "ms");
  }

  public static int main2(String[] args, Configuration conf) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Directory on HDFS containing the collapsed, properly formatted files having "
          + "one doc per line").withShortName("i").create();

    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
      abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription(
      "The path to the term-dictionary format is ... ").withShortName("d").create();

    Option dfsOpt = obuilder.withLongName("dfs").withRequired(false).withArgument(
      abuilder.withName("dfs").withMinimum(1).withMaximum(1).create()).withDescription(
      "HDFS namenode URI").withShortName("dfs").create();

    Option numTopicsOpt = obuilder.withLongName("numTopics").withRequired(true).withArgument(abuilder
        .withName("numTopics").withMinimum(1).withMaximum(1)
        .create()).withDescription("Number of topics to learn").withShortName("top").create();

    Option numTermsToPrintOpt = obuilder.withLongName("numTermsToPrint").withRequired(false).withArgument(
        abuilder.withName("numTopics").withMinimum(1).withMaximum(1).withDefault("10").create())
        .withDescription("Number of terms to print per topic").withShortName("ttp").create();

    Option outputTopicFileOpt = obuilder.withLongName("topicOutputFile").withRequired(true).withArgument(
        abuilder.withName("topicOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(term | topic)").withShortName("to").create();

    Option outputDocFileOpt = obuilder.withLongName("docOutputFile").withRequired(true).withArgument(
        abuilder.withName("docOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(topic | docid)").withShortName("do").create();

    Option alphaOpt = obuilder.withLongName("alpha").withRequired(false).withArgument(abuilder
        .withName("alpha").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(topic | document) prior").withShortName("a").create();

    Option etaOpt = obuilder.withLongName("eta").withRequired(false).withArgument(abuilder
        .withName("eta").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(term | topic)").withShortName("e").create();

    Option maxIterOpt = obuilder.withLongName("maxIterations").withRequired(false).withArgument(abuilder
        .withName("maxIterations").withMinimum(1).withMaximum(1).withDefault(10).create())
        .withDescription("Maximum number of training passes").withShortName("m").create();

    Option burnInOpt = obuilder.withLongName("burnInIterations").withRequired(false).withArgument(abuilder
        .withName("burnInIterations").withMinimum(1).withMaximum(1).withDefault(5).create())
        .withDescription("Minimum number of iterations").withShortName("b").create();

    Option convergenceOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(abuilder
        .withName("convergence").withMinimum(1).withMaximum(1).withDefault("0.0").create())
        .withDescription("Fractional rate of error to consider convergence").withShortName("c").create();

    Option minDfCtOpt = obuilder.withLongName("minDocFreq").withRequired(false).withArgument(abuilder
        .withName("minDocFreq").withMinimum(1).withMaximum(1).withDefault(2).create())
        .withDescription("Minimum document frequency (integer!) to consider in vocabulary")
        .withShortName("minDfCt").create();

    Option maxDfPctOpt = obuilder.withLongName("maxDocFreqPercentage").withRequired(false)
        .withArgument(abuilder.withName("maxDocFreqPercentage").withMinimum(1).withMaximum(1)
        .withDefault(0.5).create())
        .withDescription("Maximum percentage of documents a vocabulary term can occur in")
        .withShortName("maxDfPct").create();

    Option reInferDocTopicsOpt = obuilder.withLongName("reInferDocTopics").withRequired(false)
        .withArgument(abuilder.withName("reInferDocTopics").withMinimum(1).withMaximum(1)
        .withDefault("no").create())
        .withDescription("re-infer p(topic | doc) : [no | randstart | continue]")
        .withShortName("rdt").create();

    Option numTrainThreadsOpt = obuilder.withLongName("numTrainThreads").withRequired(false)
        .withArgument(abuilder.withName("numTrainThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to train with")
        .withShortName("ntt").create();

    Option numUpdateThreadsOpt = obuilder.withLongName("numUpdateThreads").withRequired(false)
        .withArgument(abuilder.withName("numUpdateThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to update the model with")
        .withShortName("nut").create();

    Option verboseOpt = obuilder.withLongName("verbose").withRequired(false)
        .withArgument(abuilder.withName("verbose").withMinimum(1).withMaximum(1)
        .withDefault("false").create())
        .withDescription("print verbose information, like top-terms in each topic, during iteration")
        .withShortName("v").create();


    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(numTopicsOpt)
        .withOption(numTermsToPrintOpt).withOption(alphaOpt).withOption(etaOpt)
        .withOption(maxIterOpt).withOption(burnInOpt).withOption(convergenceOpt)
        .withOption(minDfCtOpt).withOption(maxDfPctOpt).withOption(dictOpt).withOption(reInferDocTopicsOpt)
        .withOption(outputDocFileOpt).withOption(outputTopicFileOpt).withOption(dfsOpt)
        .withOption(numTrainThreadsOpt).withOption(numUpdateThreadsOpt)
        .withOption(verboseOpt).create();

    try {
      Parser parser = new Parser();

      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String inputDirString = (String) cmdLine.getValue(inputDirOpt);
      String dictDirString = (String) cmdLine.getValue(dictOpt);
      int numTopics = Integer.parseInt((String) cmdLine.getValue(numTopicsOpt));
      int numTermsToPrint = Integer.parseInt((String)cmdLine.getValue(numTermsToPrintOpt));
      double alpha = Double.parseDouble((String)cmdLine.getValue(alphaOpt));
      double eta = Double.parseDouble((String)cmdLine.getValue(etaOpt));
      int maxIterations = Integer.parseInt((String)cmdLine.getValue(maxIterOpt));
      int burnInIterations = Integer.parseInt((String)cmdLine.getValue(burnInOpt));
      double minFractionalErrorChange = Double.parseDouble((String) cmdLine.getValue(convergenceOpt));
      int minDfCt = Integer.parseInt((String)cmdLine.getValue(minDfCtOpt));
      double maxDfPct = Float.parseFloat((String)cmdLine.getValue(maxDfPctOpt));
      int numTrainThreads = Integer.parseInt((String)cmdLine.getValue(numTrainThreadsOpt));
      int numUpdateThreads = Integer.parseInt((String)cmdLine.getValue(numUpdateThreadsOpt));
      String topicOutFile = (String)cmdLine.getValue(outputTopicFileOpt);
      String docOutFile = (String)cmdLine.getValue(outputDocFileOpt);
      String reInferDocTopics = (String)cmdLine.getValue(reInferDocTopicsOpt);
      boolean verbose = Boolean.parseBoolean((String) cmdLine.getValue(verboseOpt));

      long start = System.nanoTime();
      InMemoryCollapsedVariationalBayes0 cvb0 = null;
      if(dictDirString == null) {
        Map<Integer, Map<String, Integer>> corpus = loadCorpus(inputDirString);
        logTime("text-based corpus loading", System.nanoTime() - start);
        start = System.nanoTime();
        cvb0 = new InMemoryCollapsedVariationalBayes0(corpus, numTopics, alpha, eta, minDfCt, maxDfPct);
        logTime("cvb0 init", System.nanoTime() - start);
      } else {
        if(conf.get("fs.default.name") == null) {
          String dfsNameNode = (String)cmdLine.getValue(dfsOpt);
          conf.set("fs.default.name", dfsNameNode);
        }
        String[] terms = loadDictionary(dictDirString, conf);
        logTime("dictionary loading", System.nanoTime() - start);
        start = System.nanoTime();
        Matrix corpus = loadVectors(inputDirString, conf);
        logTime("vector seqfile corpus loading", System.nanoTime() - start);
        start = System.nanoTime();
        cvb0 = new InMemoryCollapsedVariationalBayes0(corpus, terms, numTopics, alpha, eta,
            numTrainThreads, numUpdateThreads);
        logTime("cvb0 init", System.nanoTime() - start);
      }
      start = System.nanoTime();
      cvb0.setVerbose(verbose);
      double error = cvb0.iterateUntilConvergence(minFractionalErrorChange, maxIterations, burnInIterations);
      logTime("total training time", System.nanoTime() - start);

      if(reInferDocTopics.equalsIgnoreCase("randstart")) {
        cvb0.inferDocuments(0.0, 100, true);
      } else if(reInferDocTopics.equalsIgnoreCase("continue")) {
        cvb0.inferDocuments(0.0, 100, false);
      }

      start = System.nanoTime();
      cvb0.writeModel(new Path(topicOutFile));
      DistributedRowMatrixWriter.write(new Path(docOutFile), conf, cvb0.docTopicCounts);
      logTime("printTopics", System.nanoTime() - start);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
    return 0;
  }

  private static Map<Integer, Map<String, Integer>> loadCorpus(String path) throws IOException {
    List<String> lines = Resources.readLines(Resources.getResource(path), Charsets.UTF_8);
    Map<Integer, Map<String, Integer>> corpus = Maps.newHashMap();
    for(int i=0; i<lines.size(); i++) {
      String line = lines.get(i);
      Map<String, Integer> doc = Maps.newHashMap();
      for(String s : line.split(" ")) {
        s = s.replaceAll("\\W", "").toLowerCase().trim();
        if(s.length() == 0) {
          continue;
        }
        if(!doc.containsKey(s)) {
          doc.put(s, 0);
        }
        doc.put(s, doc.get(s) + 1);
      }
      corpus.put(i, doc);
    }
    return corpus;
  }

  private static String[] loadDictionary(String dictionaryPath, Configuration conf)
      throws IOException {
    Path dictionaryFile = new Path(dictionaryPath);
    List<Pair<Integer, String>> termList = Lists.newArrayList();
    int maxTermId = 0;
     // key is word value is id
    for (Pair<Writable, IntWritable> record
            : new SequenceFileIterable<Writable, IntWritable>(dictionaryFile, true, conf)) {
      termList.add(new Pair<Integer, String>(record.getSecond().get(),
          record.getFirst().toString()));
      maxTermId = Math.max(maxTermId, record.getSecond().get());
    }
    String[] terms = new String[maxTermId + 1];
    for(Pair<Integer, String> pair : termList) {
      terms[pair.getFirst()] = pair.getSecond();
    }
    return terms;
  }

  @Override
  public Configuration getConf() {
    if(super.getConf() == null) {
      setConf(new Configuration());
    }
    return super.getConf();
  }

  private static Matrix loadVectors(String vectorPathString, Configuration conf)
    throws IOException {
    Path vectorPath = new Path(vectorPathString);
    FileSystem fs = vectorPath.getFileSystem(conf);
    List<Path> subPaths = Lists.newArrayList();
    if(fs.isFile(vectorPath)) {
      subPaths.add(vectorPath);
    } else {
      for(FileStatus fileStatus : fs.listStatus(vectorPath)) {
        subPaths.add(fileStatus.getPath());
      }
    }
    List<Vector> vectorList = Lists.newArrayList();
    for(Path subPath : subPaths) {
      for(Pair<IntWritable, VectorWritable> record
          : new SequenceFileIterable<IntWritable, VectorWritable>(subPath, true, conf)) {
        vectorList.add(record.getSecond().get());
      }
    }
    int numRows = vectorList.size();
    int numCols = vectorList.get(0).size();
    return new SparseRowMatrix(new int[] {numRows, numCols},
        vectorList.toArray(new Vector[vectorList.size()]), true,
        vectorList.get(0).isSequentialAccess());
  }

  @Override public int run(String[] strings) throws Exception {
    return main2(strings, getConf());
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InMemoryCollapsedVariationalBayes0(), args);
  }
}
