package org.apache.mahout.math.stats;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class SamplerTest extends MahoutTestCase {

  @Test
  public void testDiscreteSampler() {
    Vector distribution = new DenseVector(new double[] {1, 0, 2, 3, 5, 0});
    Sampler sampler = new Sampler(RandomUtils.getRandom(1234), distribution);
    Vector sampledDistribution = distribution.like();
    int i = 0;
    while(i < 10000) {
      int index = sampler.sample();
      sampledDistribution.set(index, sampledDistribution.get(index) + 1);
      i++;
    }
    assertTrue("sampled distribution is far from the original",
        l1Dist(distribution, sampledDistribution) < 1e-2);
  }

  private double l1Dist(Vector v, Vector w) {
    return v.normalize(1d).minus(w.normalize(1)).norm(1d);
  }
}
