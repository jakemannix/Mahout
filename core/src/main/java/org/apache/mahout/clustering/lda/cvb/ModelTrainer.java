package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorIterable;

import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ModelTrainer {
  private int numTopics;
  private int numTerms;
  private TopicModel readModel;
  private TopicModel writeModel;
  private ExecutorService threadPool;
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
    threadPool = Executors.newFixedThreadPool(numTrainThreads);
  }

  public void train(VectorIterable matrix, VectorIterable docTopicCounts) {
    start();
    Iterator<MatrixSlice> docIterator = matrix.iterator();
    Iterator<MatrixSlice> docTopicIterator = docTopicCounts.iterator();
    while(docIterator.hasNext() && docTopicIterator.hasNext()) {
      Vector document = docIterator.next().vector();
      Vector topicDist = docTopicIterator.next().vector();
      train(document, topicDist, true);
    }
    stop();
  }

  public void train(Vector document, Vector docTopicCounts, boolean update) {
    threadPool.submit(new TrainerRunnable(readModel, update ? writeModel : null, document,
        docTopicCounts, new SparseRowMatrix(new int[]{numTopics, numTerms}, true)));
  }

  public void stop() {
    try {
      threadPool.shutdown();
      threadPool.awaitTermination(60, TimeUnit.SECONDS);
      writeModel.awaitTermination();
      TopicModel tmpModel = writeModel;
      writeModel = readModel;
      readModel = tmpModel;
      writeModel.reset();
    } catch (InterruptedException e) {
      //
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

    public TrainerRunnable(TopicModel readModel, TopicModel writeModel, Vector document,
        Vector docTopics, Matrix docTopicModel) {
      this.readModel = readModel;
      this.writeModel = writeModel;
      this.document = document;
      this.docTopics = docTopics;
      this.docTopicModel = docTopicModel;
    }

    @Override public void run() {
      readModel.trainDocTopicModel(document, docTopics, docTopicModel);
      if(writeModel != null) {
        writeModel.update(docTopicModel);
      }
    }
  }
}
