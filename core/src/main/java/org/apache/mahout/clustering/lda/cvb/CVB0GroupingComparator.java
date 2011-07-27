package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.RawComparator;

public class CVB0GroupingComparator implements RawComparator {

  @Override public int compare(byte[] bytes, int start, int len,
                               byte[] bytes1, int start1, int len1) {
    // just sort without the boolean: termId first, then docId.  Should return 0 if docId and termId are equal
    return CVBSortingComparator.compareNoBooleanCheck(bytes, start, len, bytes1, start1, len1);
  }

  @Override public int compare(Object x, Object y) {
    if(true)
      throw new UnsupportedOperationException("NO!");
    CVBKey k1 = (CVBKey)x;
    CVBKey k2 = (CVBKey)y;
    return k1.compareToOnlyIgnoreBoolean(k2);
  }
}
