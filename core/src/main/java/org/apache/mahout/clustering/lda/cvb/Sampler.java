package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Random;

public class Sampler {

  private Random random;

  public Sampler(Random random) {
    this.random = random;
  }

  public int sample(Vector distribution) {
    return sample(samplerFor(distribution));
  }

  private double[] samplerFor(double[] distribution) {
    return samplerFor(new DenseVector(distribution));
  }

  private double[] samplerFor(Vector vectorDistribution) {
    int size = vectorDistribution.size();
    double[] partition = new double[size];
    double norm = vectorDistribution.norm(1);
    double sum = 0;
    for(int i = 0; i < size; i++) {
      sum += (vectorDistribution.get(i) / norm);
      partition[i] = sum;
    }
    return partition;
  }

  private int sample(double[] sampler) {
    int index = Arrays.binarySearch(sampler, random.nextDouble());
    return index < 0 ? -(index+1) : index;
  }
}
