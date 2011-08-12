package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Partitioner;

public class CVB0Partitioner extends Partitioner<CVBKey, CVBTuple> {
  @Override public int getPartition(CVBKey key, CVBTuple value, int numPartitions) {
    return (((key.getDocId() + 2) * (key.getTermId() + 2) * 137) & Integer.MAX_VALUE) % numPartitions;
  }
}
