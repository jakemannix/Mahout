package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;
import org.apache.mahout.math.list.DoubleArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
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
  private boolean isReadWrite;

  private volatile double totalPerplexity;

  public ModelTrainer(TopicModel model, int numTrainThreads, int numTopics, int numTerms) {
    this(model, model, numTrainThreads, numTopics, numTerms);
  }

  public ModelTrainer(TopicModel initialReadModel, TopicModel initialWriteModel,
      int numTrainThreads, int numTopics, int numTerms) {
    this.readModel = initialReadModel;
    this.writeModel = initialWriteModel;
    this.numTrainThreads = numTrainThreads;
    this.numTopics = numTopics;
    this.numTerms = numTerms;
    isReadWrite = (initialReadModel == initialWriteModel);
    totalPerplexity = 0;
  }

  public TopicModel getReadModel() {
    return readModel;
  }

  public double getTotalPerplexity() {
    return totalPerplexity;
  }

  public void start() {
    log.info("Starting training threadpool with " + numTrainThreads + " threads");
    workQueue = new ArrayBlockingQueue<Runnable>(numTrainThreads * 10);
    threadPool = new ThreadPoolExecutor(numTrainThreads, numTrainThreads, 0, TimeUnit.SECONDS,
        workQueue);
    threadPool.allowCoreThreadTimeOut(false);
    threadPool.prestartAllCoreThreads();
  }

  public double calculatePerplexity(VectorIterable matrix, VectorIterable docTopicCounts) {
    if(totalPerplexity > 0) {
      return totalPerplexity;
    }
    Iterator<MatrixSlice> docIterator = matrix.iterator();
    Iterator<MatrixSlice> docTopicIterator = docTopicCounts.iterator();
    double perplexity = 0;
    double matrixNorm = 0;
    while(docIterator.hasNext() && docTopicIterator.hasNext()) {
      Vector document = docIterator.next().vector();
      Vector topicDist = docTopicIterator.next().vector();
      perplexity += readModel.perplexity(document, topicDist);
      matrixNorm += document.norm(1);
    }
    return perplexity / matrixNorm;
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts, double convergence) {
    train(matrix, docTopicCounts, 1, convergence);
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts, int maxDocTopicIters,
      double convergence) {
    totalPerplexity = 0;
    start();
    Iterator<MatrixSlice> docIterator = matrix.iterator();
    Iterator<MatrixSlice> docTopicIterator = docTopicCounts.iterator();
    long startTime = System.nanoTime();
    int i = 0;
    double[] times = new double[100];
    Map<Vector, Vector> batch = Maps.newHashMap();
    int numTokensInBatch = 0;
    long batchStart = System.nanoTime();
    while(docIterator.hasNext() && docTopicIterator.hasNext()) {
      i++;
      Vector document = docIterator.next().vector();
      Vector topicDist = docTopicIterator.next().vector();
      if(isReadWrite) {
        if(batch.size() < numTrainThreads) {
          batch.put(document, topicDist);
          if(log.isDebugEnabled()) {
            numTokensInBatch += document.getNumNondefaultElements();
          }
        } else {
          batchTrain(batch, true, maxDocTopicIters, convergence);
          long time = System.nanoTime();
          log.debug("trained {} docs with {} tokens, start time {}, end time {}",
              new Object[] {numTrainThreads, numTokensInBatch, batchStart, time});
          batchStart = time;
          numTokensInBatch = 0;
        }
      } else {
        long start = System.nanoTime();
        train(document, topicDist, true, maxDocTopicIters, convergence);
        if(log.isDebugEnabled()) {
          times[i % times.length] =
              ((System.nanoTime() - start)/(1e6 * document.getNumNondefaultElements()));
          if(i % 100 == 0) {
            long time = System.nanoTime() - startTime;
            log.debug("trained " + i + " documents in " + (time * 1d / 1e6) + "ms");
            if(i % 500 == 0) {
              Arrays.sort(times);
              log.debug("training took median " + times[times.length / 2] + "ms per token-instance");
            }
          }
        }
      }
    }
    stop();
  }

  public void batchTrain(Map<Vector, Vector> batch, boolean update, int numDocTopicsIters,
      double convergence) {
    while(true) {
      try {
        List<TrainerRunnable> runnables = Lists.newArrayList();
        for(Map.Entry<Vector, Vector> entry : batch.entrySet()) {
          runnables.add(new TrainerRunnable(readModel, null, entry.getKey(),
              entry.getValue(), new SparseRowMatrix(new int[]{numTopics, numTerms}, true),
              numDocTopicsIters, convergence));
        }
        threadPool.invokeAll(runnables);
        if(update) {
          for(TrainerRunnable runnable : runnables) {
            writeModel.update(runnable.docTopicModel);
          }
        }
      } catch (InterruptedException e) {
        log.warn("Interrupted during batch training, retrying!", e);
      }
    }
  }

  public void train(Vector document, Vector docTopicCounts, boolean update, int numDocTopicIters,
      double convergence) {
    while(true) {
      try {
        workQueue.put(new TrainerRunnable(readModel,
            update ? writeModel : null, document, docTopicCounts, new SparseRowMatrix(new int[]{
            numTopics, numTerms}, true), numDocTopicIters, convergence));
        return;
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to submit document to work queue: " + document, e);
      }
    }
  }

  public double calculatePerplexity(Vector document, Vector docTopicCounts, int numDocTopicIters) {
    TrainerRunnable runner =  new TrainerRunnable(readModel,
            null, document, docTopicCounts, new SparseRowMatrix(new int[]{
            numTopics, numTerms}, true), numDocTopicIters);
    return runner.call();
  }

  public static double pctDelta(DoubleArrayList list) {
    int sz = list.size();
    return Math.abs((list.get(sz - 1) - list.get(sz - 2)) / list.get(0));
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

  private class TrainerRunnable implements Runnable, Callable<Double> {
    private final TopicModel readModel;
    private final TopicModel writeModel;
    private final Vector document;
    private final Vector docTopics;
    private final Matrix docTopicModel;
    private final int maxDocTopicIters;
    private final double convergence;

    private double finalPerplexity = -1;

    public TrainerRunnable(TopicModel readModel, TopicModel writeModel, Vector document,
        Vector docTopics, Matrix docTopicModel, int maxDocTopicIters) {
      this(readModel, writeModel, document, docTopics, docTopicModel, maxDocTopicIters, Double.NaN);
    }

    public TrainerRunnable(TopicModel readModel, TopicModel writeModel, Vector document,
        Vector docTopics, Matrix docTopicModel, int maxDocTopicIters, double convergence) {
      this.readModel = readModel;
      this.writeModel = writeModel;
      this.document = document;
      this.docTopics = docTopics;
      this.docTopicModel = docTopicModel;
      this.maxDocTopicIters = maxDocTopicIters;
      this.convergence = convergence  / 10;
    }

    @Override public void run() {
      DoubleArrayList perplexities = null;
      if(!Double.isNaN(convergence)) {
        perplexities = new DoubleArrayList(maxDocTopicIters);
        perplexities.add(readModel.perplexity(document, docTopics));
      }
      int i = 0;
      while(!converged(i++, perplexities)) {
        // synchronous read-only call:
        readModel.trainDocTopicModel(document, docTopics, docTopicModel);
        if(!Double.isNaN(convergence)) {
          perplexities.add(readModel.perplexity(document, docTopics));
        }
      }
      if(writeModel != null) {
        // parallel call which is read-only on the docTopicModel, and write-only on the writeModel
        // this method does not return until all rows of the docTopicModel have been submitted
        // to write work queues
        writeModel.update(docTopicModel);
      }
      if(perplexities != null) {
        finalPerplexity = perplexities.get(perplexities.size() - 1);
        totalPerplexity += finalPerplexity;
      }
    }

    private boolean converged(int i, DoubleArrayList perplexities) {
      if(i > maxDocTopicIters) {
        return true;
      }
      if(perplexities == null) {
        return false;
      }
      if(i < 3) {
        return false;
      }
      return pctDelta(perplexities) < convergence;
    }

    @Override public Double call() {
      run();
      return finalPerplexity > 0 ? finalPerplexity : readModel.perplexity(document, docTopics);
    }
  }
}
