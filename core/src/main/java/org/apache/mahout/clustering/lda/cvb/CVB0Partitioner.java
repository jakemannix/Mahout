package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Partitioner;

public class CVB0Partitioner extends Partitioner<CVBKey, CVBTuple> {
  @Override
  public int getPartition(CVBKey key, CVBTuple value, int numPartitions) {
    int docId = key.getDocId();
    int termId = key.getTermId();
    if(termId == -1 && docId < 0) {
      return getPartition(key.getDocId(), numPartitions);
    } else {
      return (((key.getDocId() + 2) * (key.getTermId() + 2) * 137) & Integer.MAX_VALUE)
             % numPartitions;
    }
  }

  public static int getPartition(int docId, int numPartitions) {
    return ((-(docId + 1)) & Integer.MAX_VALUE) % numPartitions;
  }
}
