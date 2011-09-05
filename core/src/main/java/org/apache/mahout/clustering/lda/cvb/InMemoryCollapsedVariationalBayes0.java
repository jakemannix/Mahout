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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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

  private boolean cacheResiduals = true;

  private Map<String, Integer> termIdMap;
  private String[] terms;  // of length numTerms;

  private Vector[] corpusWeights; // length numDocs;
  private double totalCorpusWeight;
  private Vector[][] gamma; // length numDocs, each of length numTopics
  private Vector[][] gammaTimesCorpus; // length numDocs, each of length numTopics
  private double[][] docTopicCounts; // length numDocs, each of length numTopics
  private double[] docNorms; // length numDocs
  private double[][] topicTermCounts; // length numTerms, each of length numTopics

  private double[] topicCounts; // sum_a (t(x,a)) = t_x

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

  public InMemoryCollapsedVariationalBayes0(Vector[] corpus, String[] terms, int numTopics,
      double alpha, double eta) {
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    this.minDfCt = 0;
    this.maxDfPct = 1.0f;
    corpusWeights = corpus;
    numDocuments = corpus.length;
    this.terms = terms;
    numTerms = terms.length;
    termIdMap = Maps.newHashMap();
    for(int t=0; t<terms.length; t++) {
      termIdMap.put(terms[t], t);
    }
    postInitCorpus();
    initializeModel();
  }

  private void postInitCorpus() {
    totalCorpusWeight = 0;
    int numNonZero = 0;
    for(int i=0; i<numDocuments; i++) {
      Vector v = corpusWeights[i];
      double norm;
      if(v != null && (norm = v.norm(1)) != 0) {
        numNonZero += v.getNumNondefaultElements();
        totalCorpusWeight += norm;
      }
    }
    String s = "Initializing corpus with %d docs, %d terms, %d nonzero entries, total termWeight %f";
    System.out.println(String.format(s, numDocuments, numTerms, numNonZero, totalCorpusWeight));
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
    corpusWeights = new Vector[numDocuments];
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
        corpusWeights[i] = docVector;
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
        System.out.println(e.getKey() + " occurs " + df + " times, removing");
        it.remove();
      } else if(df < minDfCt) {
        it.remove();
      }
    }
    return termCounts;
  }

  private void initializeModelNoResiduals() {
    topicTermCounts = new double[numTerms][];
    for(int t = 0; t < topicTermCounts.length; t++) {
      topicTermCounts[t] = new double[numTopics];
    }
    docTopicCounts = new double[numDocuments][];
    for(int i = 0; i < numDocuments; i++) {
      docTopicCounts[i] = new double[numTopics];
    }
    docNorms = new double[numDocuments];
    topicCounts = new double[numTopics];
    Random random = new Random(1234);
    for(int i = 0; i < numDocuments; i++) {
      Vector document = corpusWeights[i];
      if(document == null) {
        continue;
      }
      Vector[] g = gamma(i);
      Vector[] gtm = gammaTimesCorpus(i);
      Iterator<Vector.Element> it = document.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        double norm = 0;
        for(int x=0; x<numTopics; x++) {
          double d = random.nextDouble();
          norm += d;
          g[x].set(e.index(), d);
        }
        for(int x=0; x<numTopics; x++) {
          double d = g[x].get(e.index()) / norm;
          g[x].set(e.index(), d);
        }
      }
      it = document.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        double[] currentTermTopicCounts = topicTermCounts[e.index()];
        for(int x=0; x<numTopics; x++) {
          double d = g[x].get(e.index()) * document.get(e.index());
          gtm[x].set(e.index(), d);
          currentTermTopicCounts[x] += d;
          docTopicCounts[i][x] += d;
        }
      }
      double di = 0;
      for(int x=0; x<numTopics; x++) {
        di += docTopicCounts[i][x];
      }
      docNorms[i] = di;
    }
  }

  private void initializeModel() {
    if(cacheResiduals) {
      initializeModelCacheResiduals();
    } else {
      initializeModelNoResiduals();
    }
  }

  private void initializeModelCacheResiduals() {
    Random random = new Random(1234);
    gamma = new Vector[numDocuments][];
    gammaTimesCorpus = new Vector[numDocuments][];
    topicTermCounts = new double[numTerms][];
    for(int t = 0; t < topicTermCounts.length; t++) {
      topicTermCounts[t] = new double[numTopics];
    }
    docTopicCounts = new double[numDocuments][];
    for(int i = 0; i < numDocuments; i++) {
      docTopicCounts[i] = new double[numTopics];
    }
    docNorms = new double[numDocuments];
    topicCounts = new double[numTopics];
    for(int i=0; i<corpusWeights.length; i++) {
      Vector document = corpusWeights[i];
      if(document == null) {
        continue;
      }
      // initialize model
      gamma[i] = new Vector[numTopics];
      gammaTimesCorpus[i] = new Vector[numTopics];
      for(int x = 0; x < numTopics; x++) {
        gamma[i][x] = new RandomAccessSparseVector(numTerms, document.getNumNondefaultElements());
        gammaTimesCorpus[i][x] = new RandomAccessSparseVector(numTerms, document.getNumNondefaultElements());
      }
      Iterator<Vector.Element> it = document.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        double norm = 0;
        for(int x=0; x<numTopics; x++) {
          double d = random.nextDouble();
          norm += d;
          gamma[i][x].set(e.index(), d);
        }
        for(int x=0; x<numTopics; x++) {
          double d = gamma[i][x].get(e.index()) / norm;
          gamma[i][x].set(e.index(), d);
        }
      }
      it = document.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        double[] currentTermTopicCounts = topicTermCounts[e.index()];
        for(int x=0; x<numTopics; x++) {
          double d = gamma[i][x].get(e.index()) * document.get(e.index());
          gammaTimesCorpus[i][x].set(e.index(), d);
          currentTermTopicCounts[x] += d;
          docTopicCounts[i][x] += d;
        }
      }
      double di = 0;
      for(int x=0; x<numTopics; x++) {
        di += docTopicCounts[i][x];
      }
      docNorms[i] = di;
    }
    for(int a = 0; a < numTerms; a++) {
      double[] currentTermTopicCounts = topicTermCounts[a];
      for(int x=0; x<numTopics; x++) {
        topicCounts[x] += currentTermTopicCounts[x];
      }
    }
  }

  private int currentTemp = -1;
  private Vector[] tempGamma;
  private int currentTempTimesCorpus = -1;
  private Vector[] tempGammaTimesCorpus;

  private Vector[] gamma(int docId) {
    if(cacheResiduals) {
      return gamma[docId];
    } else {
      if(tempGamma == null || docId != currentTemp) {
        currentTemp = docId;
        tempGamma = new Vector[numTopics];
        for(int x = 0; x < numTopics; x++) {
          tempGamma[x] = corpusWeights[docId].like();
        }
      }
      return tempGamma;
    }
  }

  private Vector[] gammaTimesCorpus(int docId) {
    if(cacheResiduals) {
      return gammaTimesCorpus[docId];
    } else {
      if(tempGammaTimesCorpus == null || docId != currentTempTimesCorpus) {
        currentTempTimesCorpus = docId;
        tempGammaTimesCorpus = new Vector[numTopics];
        for(int x = 0; x < numTopics; x++) {
          tempGammaTimesCorpus[x] = corpusWeights[docId].like();
        }
      }
      return tempGammaTimesCorpus;
    }
  }

  private void trainDocument(int docId) {
    trainDocument(docId, true);
  }

  private void trainDocument(int docId, boolean updateModel) {
    Vector document = corpusWeights[docId];
    if(document == null) {
      return;
    }
    Vector[] docModel = gamma(docId);
    Vector[] gammaTimesDocModel = gammaTimesCorpus(docId);
    double[] currentDocTopicCounts = docTopicCounts[docId];

    // update p(x|i,a) = docModel.get(a)[x] for terms a, topics x.
    Iterator<Vector.Element> it = document.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int term = e.index();
      double norm = 0;
      for(int x=0; x<numTopics; x++) {
        double d = (topicTermCounts[term][x] + eta) / (topicCounts[x] + numTerms * eta);
        d *= (currentDocTopicCounts[x] + alpha);
        docModel[x].set(term, d);
        norm += d;
      }
      for(int x = 0; x < numTopics; x++) {
        double d = docModel[x].get(term) / norm;
        docModel[x].set(term, d);
      }
      double termWeight = e.get();
      for(int x=0; x<numTopics; x++) {
        gammaTimesDocModel[x].set(term, docModel[x].get(term) * termWeight);
      }
    }
    if(!cacheResiduals && updateModel) {
      updateTopics(docId);
    }
    updateDocs(docId);
  }

  private void updateDocs(int docId) {
    Vector[] gammaTimesCorpus = gammaTimesCorpus(docId);
    docNorms[docId] = 0;
    double[] currentDocTopicCounts = docTopicCounts[docId];
    Arrays.fill(currentDocTopicCounts, 0);
    for(int x = 0; x < numTopics; x++) {
      Iterator<Vector.Element> it = gammaTimesCorpus[x].iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        docNorms[docId] += e.get();
        currentDocTopicCounts[x] += e.get();
      }
    }
  }

  private void updateTopics(int docId) {
    Vector[] gammaTimesCorpus = gammaTimesCorpus(docId);
    for(int x = 0; x < numTopics; x++) {
      Iterator<Vector.Element> it = gammaTimesCorpus[x].iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        topicTermCounts[e.index()][x] += e.get();
        topicCounts[x] += e.get();
      }
    }
  }

  private void inferDocuments(double convergence, int maxIter, boolean recalculate) {
    for(int docId = 0; docId < corpusWeights.length; docId++) {
      inferDocument(docId, convergence, maxIter, recalculate);
    }
  }

  private void inferDocument(int docId, double convergence, int maxIter, boolean recalculate) {
    double docError = error(docId);
    System.out.println(docError + " = initial Error[" + docId + "]");
    int it = 0;
    if(recalculate) {
      System.out.println("re-initializing docTopics");
      Random rand = new Random(1234 * (docId+1));
      double total = 0;
      for(int x = 0; x < numTopics; x++) {
        docTopicCounts[docId][x] = 1;
        total += docTopicCounts[docId][x];
      }
      for(int x = 0; x < numTopics; x++) {
        docTopicCounts[docId][x] /= total;
      }
    }
    double origError = docError;
    docError = Double.MAX_VALUE;
    while(docError > origError && it < maxIter) {
      trainDocument(docId, false);
      docError = error(docId);
      if(it % 25 == 0) {
        System.out.println(docError + " = e[" + it + "]");
      }
      it++;
    }
  }

  public void trainDocuments() {
    long start = System.nanoTime();
    for(int docId = 0; docId < numDocuments; docId++) {
      trainDocument(docId);
    }
    logTime("train documents", System.nanoTime() - start);
  }

  // the auxiliary gamma has been updated already in the train() step, now update docTopicCounts
  // and docNorms
  private void aggregateDocUpdates(int docId) {
    if(corpusWeights[docId] == null) {
      return;
    }
    Vector[] txia = gammaTimesCorpus[docId];
    double[] tix = new double[numTopics];
    double di = 0;
    for(int x = 0; x < numTopics; x++) {
      Iterator<Vector.Element> it = txia[x].iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        tix[x] += e.get();
        di += e.get();
      }
    }
    docTopicCounts[docId] = tix;
    docNorms[docId] = di;
  }

  private void aggregateUpdates() {
    long time = System.nanoTime();
    for(int docId = 0; docId < numDocuments; docId++) {
      aggregateDocUpdates(docId);
    }
    logTime("updateDocuments", System.nanoTime() - time);
    time = System.nanoTime();
    Arrays.fill(topicCounts, 0d);
    for(int term = 0; term < numTerms; term++) {
      Arrays.fill(topicTermCounts[term], 0d);
    }
    for(int docId = 0; docId < corpusWeights.length; docId++) {
      if(corpusWeights[docId] == null) {
        continue;
      }
      for(int x = 0; x < numTopics; x++) {
        Vector g = gammaTimesCorpus[docId][x];
        Iterator<Vector.Element> it = g.iterateNonZero();
        while(it.hasNext()) {
          Vector.Element e = it.next();
          topicTermCounts[e.index()][x] += e.get();
          topicCounts[x] += e.get();
        }
      }
    }
    logTime("udpateTerms", System.nanoTime() - time);
  }

  private double error(int docId) {
    Vector docTermCounts = corpusWeights[docId];
    if(docTermCounts == null) {
      return 0;
    } else {
      Vector expectedDocTermCounts = expectedDocumentCounts(docId);
      return expectedDocTermCounts.minus(docTermCounts).norm(1);
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

  private Vector expectedDocumentCounts(int docId) {
    // compute p(topic | docId) for all topics
    double[] pTopicDoc = new double[numTopics];
    double[] docTopicCount = docTopicCounts[docId];
    double expectedDocLength = docNorms[docId];
    for(int x=0; x<numTopics; x++) {
      pTopicDoc[x] = docTopicCount[x] / expectedDocLength;
    }

    Vector expectedVector = corpusWeights[docId].like();
    Vector[] docModel = gamma(docId);

    for(int x = 0; x < numTopics; x++) {
      Vector docTopicModel = docModel[x];
      Iterator<Vector.Element> it = docTopicModel.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        int term = e.index();
        double pTermTopic = topicTermCounts[term][x] / topicCounts[x];
        expectedVector.set(term,
            expectedVector.get(term) + pTermTopic * pTopicDoc[x] / expectedDocLength);
      }
    }

    return expectedVector.times(corpusWeights[docId].norm(1) / expectedVector.norm(1));
  }

  public double iterate() {
    trainDocuments();
    double error = 1;
    if(cacheResiduals) {
      aggregateUpdates();
      error = error();
    }
    System.out.println(error + " = error");
    return error;
  }

  public double iterateUntilConvergence(double minFractionalErrorChange, int maxIterations, int minIter) {
    double fractionalChange = Double.MAX_VALUE;
    int iter = 0;
    double oldError = 0;
    while(iter < minIter) {
      oldError = iterate();
      iter++;
    }
    double newError = 0;
    while(iter < maxIterations && fractionalChange > minFractionalErrorChange) {
      newError = iterate();
      iter++;
      fractionalChange = Math.abs(newError - oldError) / oldError;
      System.out.println(fractionalChange + " = fractionalChange");
      oldError = newError;
    }
    if(iter < maxIterations) {
      System.out.println(String.format("Converged! fractional error change: %f, error %f",
          fractionalChange, newError));
    } else {
      System.out.println(String.format("Reached max iteration count (%d), fractional error change: %f, error: %f",
          maxIterations, fractionalChange, newError));
    }
    return newError;
  }

  public void writeTopicModel(int numTerms, Path outputPath) {
    Map<Integer, Map<String, Double>> pTopicTerm = Maps.newHashMap();
    for(int term = 0; term < topicTermCounts.length; term++) {
      double[] topicWordCount = topicTermCounts[term].clone(); // count of topic assignments for this term
      for(int x=0; x<numTopics; x++) {
        topicWordCount[x] /= topicCounts[x]; // c(x, t) / c(x) = % of topic x which is t.
        if(!pTopicTerm.containsKey(x)) {
          pTopicTerm.put(x, Maps.<String, Double>newHashMap());
        }
        if(!pTopicTerm.get(x).containsKey(terms[term])) {
          pTopicTerm.get(x).put(terms[term], 0d);
        }
        // p(x, t) += topicWordCount(x)
        pTopicTerm.get(x).put(terms[term], pTopicTerm.get(x).get(terms[term]) + topicWordCount[x]);
      }
    }
    Map<Integer, List<Pair<String, Double>>> topTopicTerms = Maps.newHashMap();
    try {
      FileSystem fs = FileSystem.get(getConf());
      if(fs.exists(outputPath) && !fs.getFileStatus(outputPath).isDir()) {
        fs.delete(outputPath, true);
      }
      if(!fs.exists(outputPath)) {
        fs.mkdirs(outputPath);
      }
    } catch (IOException e) {
      throw new RuntimeException(e); // TODO cleanup
    }
    for(int x=0; x<numTopics; x++) {
      List<Pair<String,Double>> topTerms = Lists.newArrayList();
      for(Map.Entry<String,Double> topicTermEntry : pTopicTerm.get(x).entrySet()) {
        topTerms.add(new Pair<String, Double>(topicTermEntry.getKey(), topicTermEntry.getValue()));
      }
      Collections.sort(topTerms, new Comparator<Pair<String, Double>>() {
        @Override public int compare(Pair<String, Double> a, Pair<String, Double> b) {
          return Double.compare(b.getSecond(), a.getSecond());
        }
      });
      log.info("Writing topic (" + x + ") to " + outputPath);
      SequenceFile.Writer writer = null;
      try {
        writer = new SequenceFile.Writer(FileSystem.get(getConf()), getConf(),
            new Path(outputPath, "" + x), Text.class, DoubleWritable.class);
        Text text = new Text();
        DoubleWritable value = new DoubleWritable();
        for(int i=0; i<numTerms && i<topTerms.size(); i++) {
          Pair<String, Double> pair = topTerms.get(i);
          text.set(pair.getFirst());
          value.set(pair.getSecond());
          writer.append(text, value);
        }
      } catch (IOException ioe) {
        
      } finally {
        if(writer != null) {
          try {
            writer.close();
          } catch (IOException e) {
            // ignore
          }
        }
      }
    }
  }

  private void writeDocTopics(Path outputPath) {
    SequenceFile.Writer writer = null;
    try {
      FileSystem fs = FileSystem.get(getConf());
      if(fs.exists(outputPath)) {
        fs.delete(outputPath, true);
      }
      IntWritable key = new IntWritable();
      VectorWritable value = new VectorWritable();
      writer = new SequenceFile.Writer(fs, getConf(), outputPath,
          IntWritable.class, VectorWritable.class);
      for(int docId = 0; docId < docTopicCounts.length; docId++) {
        Vector topicVector = new DenseVector(docTopicCounts[docId], true);
        double norm = topicVector.zSum();
        if(norm > 0) {
          topicVector.assign(Functions.mult(1/norm));
        }
        key.set(docId);
        value.set(topicVector);
        writer.append(key, value);
      }
    } catch (IOException e) {
      throw new RuntimeException(e); // TODO
    } finally {
      if(writer != null) {
        try {
          writer.close();
        } catch (IOException e) {
          // ignore
        }
      }
    }
  }

  private static final void logTime(String label, long nanos) {
    System.out.println(label + " time: " + (double)(nanos)/1e6 + "ms");
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

    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(numTopicsOpt)
        .withOption(numTermsToPrintOpt).withOption(alphaOpt).withOption(etaOpt)
        .withOption(maxIterOpt).withOption(burnInOpt).withOption(convergenceOpt)
        .withOption(minDfCtOpt).withOption(maxDfPctOpt).withOption(dictOpt).withOption(reInferDocTopicsOpt)
        .withOption(outputDocFileOpt).withOption(outputTopicFileOpt).withOption(dfsOpt).create();

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
      String topicOutFile = (String)cmdLine.getValue(outputTopicFileOpt);
      String docOutFile = (String)cmdLine.getValue(outputDocFileOpt);
      String reInferDocTopics = (String)cmdLine.getValue(reInferDocTopicsOpt);

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
        Vector[] corpus = loadVectors(inputDirString, conf);
        logTime("vector seqfile corpus loading", System.nanoTime() - start);
        start = System.nanoTime();
        cvb0 = new InMemoryCollapsedVariationalBayes0(corpus, terms, numTopics, alpha, eta);
        logTime("cvb0 init", System.nanoTime() - start);
      }
      start = System.nanoTime();
      double error = cvb0.iterateUntilConvergence(minFractionalErrorChange, maxIterations, burnInIterations);
      logTime("total training time", System.nanoTime() - start);

      if(reInferDocTopics.equalsIgnoreCase("randstart")) {
        cvb0.inferDocuments(0.0, 10000, true);
      } else if(reInferDocTopics.equalsIgnoreCase("continue")) {
        cvb0.inferDocuments(0.0, 10000, false);
      }

      start = System.nanoTime();
      cvb0.writeTopicModel(numTermsToPrint, new Path(topicOutFile));
      cvb0.writeDocTopics(new Path(docOutFile));
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

  private static Vector[] loadVectors(String vectorPathString, Configuration conf)
    throws IOException {
    Path vectorPath = new Path(vectorPathString);
    List<Vector> vectorList = Lists.newArrayList();
    for(Pair<IntWritable, VectorWritable> record
        : new SequenceFileIterable<IntWritable, VectorWritable>(vectorPath, true, conf)) {
      vectorList.add(record.getSecond().get());
    }
    return vectorList.toArray(new Vector[vectorList.size()]);
  }

  @Override public int run(String[] strings) throws Exception {
    return main2(strings, getConf());
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InMemoryCollapsedVariationalBayes0(), args);
  }
}
