package org.apache.mahout.clustering.lda;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Resources;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
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

public class CollapsedVariationalBayes0 {
  private static final Logger log = LoggerFactory.getLogger(CollapsedVariationalBayes0.class);

  private int numTopics;
  private int numTerms;
  private int numDocuments;
  private double alpha;
  private double eta;
  private int minDfCt;
  private float maxDfPct;

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

  public CollapsedVariationalBayes0(Map<Integer, Map<String, Integer>> corpus,
      int numTopics, double alpha, double eta, int minDfCt, float maxDfPct) {
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    this.minDfCt = minDfCt;
    this.maxDfPct = maxDfPct;
    initializeCorpusWeights(corpus);
    initializeModel();
  }

  public CollapsedVariationalBayes0(Vector[] corpus, String[] terms,
      int numTopics, double alpha, double eta) {
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
        numNonZero++;
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

  private void initializeModel() {
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

  private void trainDocument(int docId) {
    Vector document = corpusWeights[docId];
    if(document == null) {
      return;
    }
    Vector[] docModel = gamma[docId];
    Vector[] gammaTimesDocModel = gammaTimesCorpus[docId];
    double[] currentDocTopicCounts = docTopicCounts[docId];

    // update p(x|i,a) = docModel.get(a)[x] for terms a, topics x.
    Iterator<Vector.Element> it = document.iterateNonZero();
    while(it.hasNext()) {
      Vector.Element e = it.next();
      int term = e.index();
      double norm = 0;
      // double[] docTermModel = new double[numTopics];
      // newDocModel.put(term, docTermModel);
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
    Vector[] docModel = gamma[docId];

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
    aggregateUpdates();
    double error = error();
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

  public void printTopics(int numTerms) {
    Map<Integer, Map<String, Double>> pTopicTerm = Maps.newHashMap();
    for(int term = 0; term < topicTermCounts.length; term++) {
      double[] topicWordCount = topicTermCounts[term].clone();
      for(int x=0; x<numTopics; x++) {
        topicWordCount[x] /= topicCounts[x];
        if(!pTopicTerm.containsKey(x)) {
          pTopicTerm.put(x, Maps.<String, Double>newHashMap());
        }
        if(!pTopicTerm.get(x).containsKey(terms[term])) {
          pTopicTerm.get(x).put(terms[term], 0d);
        }
        pTopicTerm.get(x).put(terms[term], pTopicTerm.get(x).get(terms[term]) + topicWordCount[x]);
      }
    }
    Map<Integer, List<Pair<String, Double>>> topTopicTerms = Maps.newHashMap();
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
      System.out.println("Topic (" + x + ")");
      for(int i=0; i<numTerms && i<topTerms.size(); i++) {
        System.out.println("  " + topTerms.get(i).getFirst() + " : " + topTerms.get(i).getSecond());
      }
    }
  }

  private static final void logTime(String label, long nanos) {
    System.out.println(label + " time: " + (double)(nanos)/1e6 + "ms");
  }

  /**
   * usage: [java invoc] inputFile numTopics numTermsToPrint \
   * [alpha eta maxIter burnIn minFractionalChange minDfCt maxDfPct]
   * @param args
   * @throws IOException
   */
  public static void main(String[] args) throws IOException {
    // TODO: get these from args!
    if(args.length < 3) {
      System.out.println("usage: [java invoc] inputFile numTopics numTermsToPrint"
                         + "[alpha eta maxIter burnIn minFractionalChange minDfCt maxDfPct]");
      System.exit(1);
    }
    int numTopics = Integer.parseInt(args[1]);
    long start = System.nanoTime();
    List<String> lines = Resources.readLines(Resources.getResource(args[0]), Charsets.UTF_8);
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
    logTime("corpus loading", System.nanoTime() - start);
    boolean userConfigured = args.length == 10;
    if(args.length > 3 && !userConfigured) {
      System.out.println("usage: [java invoc] inputFile numTopics numTermsToPrint"
                         + "[alpha eta maxIter burnIn minFractionalChange minDfCt maxDfPct]");
      System.exit(1);
    }
    int numTermsToPrint = userConfigured ? Integer.parseInt(args[2]) : 10;
    double alpha = userConfigured ? Double.parseDouble(args[3]) : 0.1;
    double eta = userConfigured ? Double.parseDouble(args[4]) : 0.1;
    int maxIterations = userConfigured ? Integer.parseInt(args[5]) : 500;
    int burnInIterations = userConfigured ? Integer.parseInt(args[6]) : 10;
    float minFractionalErrorChange = userConfigured ? Float.parseFloat(args[7]) : 0f;
    int minDfCt = userConfigured ? Integer.parseInt(args[8]) : 2;
    float maxDfPct = userConfigured ? Float.parseFloat(args[9]) : 0.5f;

    start = System.nanoTime();
    CollapsedVariationalBayes0 cvb0 = new CollapsedVariationalBayes0(corpus, numTopics, alpha, eta,
        minDfCt, maxDfPct);
    logTime("cvb0 initialization ", System.nanoTime() - start);
    double error = cvb0.iterateUntilConvergence(minFractionalErrorChange, maxIterations, burnInIterations);
    start = System.nanoTime();
    cvb0.printTopics(numTermsToPrint);
    logTime("printTopics", System.nanoTime() - start);
  }
  
}
