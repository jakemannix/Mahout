package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class CVBSortingComparator extends WritableComparator {
  protected CVBSortingComparator() {
    super(CVBKey.class);
  }

  public int compare(WritableComparable x, WritableComparable y) {
    CVBKey k1 = (CVBKey)x;
    CVBKey k2 = (CVBKey)y;
    return k1.compareTo(k2);
  }
}
