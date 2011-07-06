package org.apache.mahout.clustering.lda;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Resources;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class CollapsedVariationalBayes0 {

  private int numTopics;
  private int numTerms;
  private int numDocuments;
  private double alpha;
  private double eta;
  private int corpusSize;

  private Map<String, Integer> termIdMap;
  private String[] terms;  // of length numTerms;
  private double[] termWeights;  // length numTerms;

  private Vector[] corpusWeights; // length numDocs;
  private Vector[][] gamma; // length numDocs, each of length numTopics
  private Vector[][] gammaTimesCorpus; // length numDocs, each of length numTopics
  private double[][] docTopicCounts; // length numDocs, each of length numTopics
  private double[] docNorms; // length numDocs
  private double[][] topicTermCounts; // length numTerms, each of length numTopics

  private double[] topicCounts; // sum_a (t(x,a)) = t_x

  public CollapsedVariationalBayes0(Map<Integer, Map<String, Integer>> corpus,
      int numTopics, double alpha, double eta) {
    initializeCorpusWeights(corpus);
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    initialize();
  }

  private void initializeCorpusWeights(Map<Integer, Map<String, Integer>> corpus) {
    numDocuments = corpus.size();
    Map<String, Integer> termCounts = termCount(corpus);
    terms = termCounts.keySet().toArray(new String[termCounts.size()]);
    numTerms = terms.length;
    termWeights = new double[terms.length];
    for(int t=0; t<terms.length; t++) {
      // Calculate the iCF (inverse *collection* frequency, akin to the inverseDocumentFrequency)
      termWeights[t] = Math.log(corpusSize / (1 + termCounts.get(terms[t])));
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
        int termId = termIdMap.get(e.getKey());
        docVector.set(termId, e.getValue() * termWeights[termId]);
      }
      corpusWeights[i] = docVector;
    }
  }

  private Map<String, Integer> termCount(Map<Integer, Map<String, Integer>> corpus) {
    Map<String, Integer> termCounts = Maps.newHashMap();
    for(int docId : corpus.keySet()) {
      for(Map.Entry<String, Integer> e : corpus.get(docId).entrySet()) {
        String term = e.getKey();
        int count = e.getValue();
        if(!termCounts.containsKey(term)) {
          termCounts.put(term, 0);
        }
        termCounts.put(term, termCounts.get(term) + count);
        corpusSize += count;
      }
    }
    return termCounts;
  }

  private void initialize() {
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
    for(int docId = 0; docId < numDocuments; docId++) {
      trainDocument(docId);
    }
  }

  // the auxiliary gamma has been updated already in the train() step, now update docTopicCounts
  // and docNorms
  private void aggregateDocUpdates(int docId) {
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

  private double[] aggregateTermUpdates(int term) {
    double[] topicCounts = topicTermCounts[term];
    Arrays.fill(topicCounts, 0d);
    for(int docId = 0; docId < corpusWeights.length; docId++) {
      for(int x = 0; x < numTopics; x++) {
        Vector g = gammaTimesCorpus[docId][x];
        topicCounts[x] += g.get(term);
      }
    }
    return topicCounts;
  }

  private void aggregateUpdates() {
    for(int docId = 0; docId < numDocuments; docId++) {
      aggregateDocUpdates(docId);
    }
    Arrays.fill(topicCounts, 0d);
    for(int term = 0; term < topicTermCounts.length; term++) {
      double[] termTopicCounts = aggregateTermUpdates(term);
      for(int x=0; x<numTopics; x++) {
        topicCounts[x] += termTopicCounts[x];
      }
    }
  }

  private double error(int docId) {
    Vector docTermCounts = corpusWeights[docId];
    Vector expectedDocTermCounts = expectedDocumentCounts(docId);
    return expectedDocTermCounts.minus(docTermCounts).norm(1);
  }

  private double error() {
    double error = 0;
    for(int docId = 0; docId < numDocuments; docId++) {
      error += error(docId);
    }
    return error / corpusSize;
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

  /**
   * usage: [java invoc] inputFile numTopics numTermsToPrint [alpha eta maxIter burnIn minFractionalChange]
   * @param args
   * @throws IOException
   */
  public static void main(String[] args) throws IOException {
    // TODO: get these from args!
    if(args.length < 3) {
      System.out.println("usage: [java invoc] inputFile numTopics numTermsToPrint"
                         + "[alpha eta maxIter burnIn minFractionalChange]");
      System.exit(1);
    }
    int numTopics = Integer.parseInt(args[1]);
    List<String> lines = Resources.readLines(Resources.getResource(args[0]), Charsets.UTF_8);
    Map<Integer, Map<String, Integer>> corpus = Maps.newHashMap();
    for(int i=0; i<lines.size(); i++) {
      String line = lines.get(i);
      Map<String, Integer> doc = Maps.newHashMap();
      for(String s : line.split(" ")) {
        if(!doc.containsKey(s)) {
          doc.put(s, 0);
        }
        doc.put(s, doc.get(s) + 1);
      }
      corpus.put(i, doc);
    }
    boolean userConfigured = args.length == 8;
    if(args.length > 3 && !userConfigured) {
      System.out.println("usage: [java invoc] inputFile numTopics numTermsToPrint"
                         + "[alpha eta maxIter burnIn minFractionalChange]");
      System.exit(1);
    }
    int numTermsToPrint = userConfigured ? Integer.parseInt(args[2]) : 10;
    double alpha = userConfigured ? Double.parseDouble(args[3]) : 0.1;
    double eta = userConfigured ? Double.parseDouble(args[4]) : 0.1;
    int maxIterations = userConfigured ? Integer.parseInt(args[5]) : 500;
    int burnInIterations = userConfigured ? Integer.parseInt(args[6]) : 10;
    float minFractionalErrorChange = userConfigured ? Float.parseFloat(args[7]) : 0f;

    CollapsedVariationalBayes0 cvb0 = new CollapsedVariationalBayes0(corpus, numTopics, alpha, eta);
    cvb0.initialize();
    double error = cvb0.iterateUntilConvergence(minFractionalErrorChange, maxIterations, burnInIterations);
    cvb0.printTopics(numTermsToPrint);
  }
  
}
