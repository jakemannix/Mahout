package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparator;

/**
 * serialized format: bytes:
 * [_ _ _ _][_ _ _ _][_][_ _ _ _]
 *   termId    docId  b  branch
 */
public class CVBSortingComparator extends WritableComparator implements RawComparator {

  protected CVBSortingComparator() {
    super(CVBKey.class);
  }

  @Override public int compare(byte[] bytes, int start, int len, byte[] bytes1, int start1, int len1) {
    int result = compareNoBooleanCheck(bytes, start, len, bytes1, start1, len1);
    // if they have different docId/termId, return that result
    if(result != 0) {
      return result;
    } else {
      // same docId and termId, return reverse sorting  of the boolean as a byte (1 == true, 0 == false)
      return bytes1[start1 + 8] - bytes[start + 8];
    }
  }

  public static AggregationBranch getBranch(byte[] bytes, int offset) {
    int i = readInt(bytes, offset) - 1;
    return i < 0 ? null : AggregationBranch.values()[i];
  }

  public static int compareNoBooleanCheck(byte[] bytes, int start, int len,
                                          byte[] bytes1, int start1, int len1) {
    int termId = readInt(bytes, start);
    int termId1 = readInt(bytes1, start1);

    int docId = readInt(bytes, start + 4);
    int docId1 = readInt(bytes1, start1 + 4);

    int result = compare(docId, docId1);

    // if docIds are different, return reverse
    if(result != 0) {
      return result;
    }
    // termIds are different, return reverse sorting of docId
    result = compare(termId, termId1);
    return result;
  }

  public static int compare(int x, int y) {
    return (x < y) ? -1 : ((x == y) ? 0 : 1);
  }

  @Override public int compare(Object x, Object y) {
    CVBKey k1 = (CVBKey)x;
    CVBKey k2 = (CVBKey)y;
    return k1.compareTo(k2);
  }
}
