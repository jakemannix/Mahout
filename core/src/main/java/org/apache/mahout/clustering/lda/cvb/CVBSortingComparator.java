package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.RawComparator;

/**
 * serialized format: bytes:
 * [_ _ _ _][_ _ _ _][_][_ _ _ _]
 *   termId    docId  b  branch
 */
public class CVBSortingComparator implements RawComparator {

  @Override public int compare(byte[] bytes, int start, int len, byte[] bytes1, int start1, int len1) {
    int result = compareNoBooleanCheck(bytes, start, len, bytes1, start1, len1);
    // if they have different docId/termId, return that result
    if(result != 0) {
      return result;
    } else {
      // same docId and termId, return reverse sorting  of the boolean as a byte (1 == true, 0 == false)
      return - new Byte(bytes[start + 8]).compareTo(bytes1[start1 + 8]);
    }
  }

  public static AggregationBranch getBranch(byte[] bytes, int offset) {
    int i = extract(bytes, offset) - 1;
    return i < 0 ? null : AggregationBranch.values()[i];
  }

  public static int compareNoBooleanCheck(byte[] bytes, int start, int len,
                                          byte[] bytes1, int start1, int len1) {
    int termId = extract(bytes, start);
    int termId1 = extract(bytes1, start1);
    int result = new Integer(termId).compareTo(termId1);
    // if termIds are different, return reverse 
    if(result != 0) {
      return -result;
    }
    int docId = extract(bytes, start + 4);
    int docId1 = extract(bytes1, start1 + 4);
    // termIds are different, return reverse sorting of docId
    result = - new Integer(docId).compareTo(docId1);
    return result;
  }

  private static int extract(byte[] bytes, int offset) {
    return ((bytes[offset] << 24) |
            (bytes[offset + 1] << 16) |
            (bytes[offset + 2] << 8) |
            bytes[offset + 3]);
  }

  @Override public int compare(Object x, Object y) {
    CVBKey k1 = (CVBKey)x;
    CVBKey k2 = (CVBKey)y;
    return k1.compareTo(k2);
  }
}
