package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class ModelTrainer {
  private static final Logger log = LoggerFactory.getLogger(ModelTrainer.class);
  private int numTopics;
  private int numTerms;
  private TopicModel readModel;
  private TopicModel writeModel;
  private ThreadPoolExecutor threadPool;
  private BlockingQueue<Runnable> workQueue;
  private int numTrainThreads;

  public ModelTrainer(TopicModel initialReadModel, TopicModel initialWriteModel,
      int numTrainThreads, int numTopics, int numTerms) {
    this.readModel = initialReadModel;
    this.writeModel = initialWriteModel;
    this.numTrainThreads = numTrainThreads;
    this.numTopics = numTopics;
    this.numTerms = numTerms;
  }

  public TopicModel getReadModel() {
    return readModel;
  }

  public void start() {
    log.info("Starting training threadpool with " + numTrainThreads + " threads");
    workQueue = new ArrayBlockingQueue<Runnable>(numTrainThreads * 10);
    threadPool = new ThreadPoolExecutor(numTrainThreads, numTrainThreads, 0, TimeUnit.SECONDS,
        workQueue);
    threadPool.allowCoreThreadTimeOut(false);
    threadPool.prestartAllCoreThreads();
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts) {
    train(matrix, docTopicCounts, 1);
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts, int numDocTopicIters) {
    start();
    Iterator<MatrixSlice> docIterator = matrix.iterator();
    Iterator<MatrixSlice> docTopicIterator = docTopicCounts.iterator();
    long startTime = System.nanoTime();
    int i = 0;
    double[] times = new double[100];
    while(docIterator.hasNext() && docTopicIterator.hasNext()) {
      i++;
      Vector document = docIterator.next().vector();
      Vector topicDist = docTopicIterator.next().vector();
      long start = System.nanoTime();
      train(document, topicDist, true, numDocTopicIters);
      times[i % times.length] = ((System.nanoTime() - start)/(1e6 * document.getNumNondefaultElements()));
      if(i % 100 == 0) {
        long time = System.nanoTime() - startTime;
        log.info("trained " + i + " documents in " + (time * 1d / 1e6) + "ms");
        if(i % 500 == 0) {
          Arrays.sort(times);
          log.info("training took median " + times[times.length / 2] + "ms per token-instance");
        }
      }
    }
    stop();
  }

  public void train(Vector document, Vector docTopicCounts, boolean update, int numDocTopicIters) {
    while(true) {
      try {
        workQueue.put(new TrainerRunnable(readModel,
            update ? writeModel : null, document, docTopicCounts, new SparseRowMatrix(new int[]{
            numTopics, numTerms}, true), numDocTopicIters));
        return;
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to submit document to work queue: " + document, e);
      }
    }
  }

  public void stop() {
    long startTime = System.nanoTime();
    log.info("Initiating stopping of training threadpool");
    try {
      threadPool.shutdown();
      if(!threadPool.awaitTermination(60, TimeUnit.SECONDS)) {
        log.warn("Threadpool timed out on await termination - jobs still running!");
      }
      long newTime = System.nanoTime();
      log.info("threadpool took: " + (newTime - startTime)*1d/1e6 + "ms");
      startTime = newTime;
      writeModel.awaitTermination();
      newTime = System.nanoTime();
      log.info("writeModel.awaitTermination() took " + (newTime - startTime)*1d/1e6 + "ms");
      TopicModel tmpModel = writeModel;
      writeModel = readModel;
      readModel = tmpModel;
      writeModel.reset();
    } catch (InterruptedException e) {
      log.error("Interrupted shutting down!", e);
    }
  }

  public void persist(Path outputPath) throws IOException {
    readModel.persist(outputPath, true);
  }

  private static class TrainerRunnable implements Runnable {
    private final TopicModel readModel;
    private final TopicModel writeModel;
    private final Vector document;
    private final Vector docTopics;
    private final Matrix docTopicModel;
    private final int numDocTopicIters;

    public TrainerRunnable(TopicModel readModel, TopicModel writeModel, Vector document,
        Vector docTopics, Matrix docTopicModel, int numDocTopicIters) {
      this.readModel = readModel;
      this.writeModel = writeModel;
      this.document = document;
      this.docTopics = docTopics;
      this.docTopicModel = docTopicModel;
      this.numDocTopicIters = numDocTopicIters;
    }

    @Override public void run() {
      for(int i = 0; i < numDocTopicIters; i++) {
        // synchronous read-only call:
        readModel.trainDocTopicModel(document, docTopics, docTopicModel);
      }
      if(writeModel != null) {
        // parallel call which is read-only on the docTopicModel, and write-only on the writeModel
        // this method does not return until all rows of the docTopicModel have been submitted
        // to write work queues
        writeModel.update(docTopicModel);
      }
    }
  }
}
