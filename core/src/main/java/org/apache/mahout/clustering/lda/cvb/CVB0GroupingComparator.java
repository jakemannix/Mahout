package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparator;

public class CVB0GroupingComparator extends WritableComparator {
  public CVB0GroupingComparator() {
    super(CVBKey.class);
  }

  @Override
  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
    /*
     * Sort without the boolean: (1) termId, desc (2) docId, desc. Should return 0 if docId and
     * termId are equal.
     */
    return CVBSortingComparator.compareNoBooleanCheck(b1, s1, l1, b2, s2, l2);
  }

  @Override
  public int compare(Object x, Object y) {
    throw new UnsupportedOperationException("Binary comparison should be used");
  }
}
