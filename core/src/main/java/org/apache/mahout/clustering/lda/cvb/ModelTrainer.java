package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
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

  public double calculatePerplexity(VectorIterable matrix, VectorIterable docTopicCounts) {
    return calculatePerplexity(matrix, docTopicCounts, 0);
  }

  public double calculatePerplexity(VectorIterable matrix, VectorIterable docTopicCounts,
      double testFraction) {
    Iterator<MatrixSlice> docIterator = matrix.iterator();
    Iterator<MatrixSlice> docTopicIterator = docTopicCounts.iterator();
    double perplexity = 0;
    double matrixNorm = 0;
    while(docIterator.hasNext() && docTopicIterator.hasNext()) {
      MatrixSlice docSlice = docIterator.next();
      MatrixSlice topicSlice = docTopicIterator.next();
      int docId = docSlice.index();
      Vector document = docSlice.vector();
      Vector topicDist = topicSlice.vector();
      if(testFraction == 0 || docId % ((int)1/testFraction) == 0) {
        trainSync(document, topicDist, false, 10);
        perplexity += readModel.perplexity(document, topicDist);
        matrixNorm += document.norm(1);
      }
    }
    return perplexity / matrixNorm;
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts, int numDocTopicIters) {
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
          batchTrain(batch, true, numDocTopicIters);
          long time = System.nanoTime();
          log.debug("trained {} docs with {} tokens, start time {}, end time {}",
              new Object[] {numTrainThreads, numTokensInBatch, batchStart, time});
          batchStart = time;
          numTokensInBatch = 0;
        }
      } else {
        long start = System.nanoTime();
        train(document, topicDist, true, numDocTopicIters);
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

  public void batchTrain(Map<Vector, Vector> batch, boolean update, int numDocTopicsIters) {
    while(true) {
      try {
        List<TrainerRunnable> runnables = Lists.newArrayList();
        for(Map.Entry<Vector, Vector> entry : batch.entrySet()) {
          runnables.add(new TrainerRunnable(readModel, null, entry.getKey(),
              entry.getValue(), new SparseRowMatrix(numTopics, numTerms, true),
              numDocTopicsIters));
        }
        threadPool.invokeAll(runnables);
        if(update) {
          for(TrainerRunnable runnable : runnables) {
            writeModel.update(runnable.docTopicModel);
          }
        }
        break;
      } catch (InterruptedException e) {
        log.warn("Interrupted during batch training, retrying!", e);
      }
    }
  }

  public void train(Vector document, Vector docTopicCounts, boolean update, int numDocTopicIters) {
    while(true) {
      try {
        workQueue.put(new TrainerRunnable(readModel,
            update ? writeModel : null, document, docTopicCounts, new SparseRowMatrix(
            numTopics, numTerms, true), numDocTopicIters));
        return;
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to submit document to work queue: " + document, e);
      }
    }
  }

  public void trainSync(Vector document, Vector docTopicCounts, boolean update,
      int numDocTopicIters) {
    new TrainerRunnable(readModel,
            update ? writeModel : null, document, docTopicCounts, new SparseRowMatrix(
            numTopics, numTerms, true), numDocTopicIters).run();
  }

  public double calculatePerplexity(Vector document, Vector docTopicCounts, int numDocTopicIters) {
    TrainerRunnable runner =  new TrainerRunnable(readModel,
            null, document, docTopicCounts, new SparseRowMatrix(
            numTopics, numTerms, true), numDocTopicIters);
    return runner.call();
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

  private static class TrainerRunnable implements Runnable, Callable<Double> {
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

    @Override public Double call() {
      run();
      return readModel.perplexity(document, docTopics);
    }
  }
}
