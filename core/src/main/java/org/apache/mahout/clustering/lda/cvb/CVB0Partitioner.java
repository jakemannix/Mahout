package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Partitioner;

public class CVB0Partitioner extends Partitioner<CVBKey, CVBTuple> {
  @Override public int getPartition(CVBKey key, CVBTuple value, int numPartitions) {
    return ((key.getDocId() + 1) * (key.getTermId() + 1) * 137) % numPartitions;
  }
}
