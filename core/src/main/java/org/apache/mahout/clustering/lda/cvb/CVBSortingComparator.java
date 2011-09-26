package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparator;

/**
 * serialized format: bytes:
 * [_ _ _ _][_ _ _ _][_][_ _ _ _]
 *   termId    docId  b  branch
 */
public class CVBSortingComparator extends WritableComparator {
  /**
   * @param x
   * @param y
   * @return -1 if x is less than y, 1 if x is greater than y, and 0 if x and y are equal.
   */
  public static int compare(int x, int y) {
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
  }

  /**
   * Sorts raw CVBKey bytes by (1) termId, desc (2) docId, desc.
   */
  public static int compareNoBooleanCheck(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
    // compare termIds
    int c = -compare(readInt(b1, s1), readInt(b2, s2));
    if (c != 0) {
      return c;
    }

    // advance start offsets
    s1 += 4;
    s2 += 4;

    // compare docIds
    return -compare(readInt(b1, s1), readInt(b2, s2));
  }

  protected CVBSortingComparator() {
    super(CVBKey.class);
  }

  /**
   * Sorts raw CVBKey bytes by (1) termId, desc (2) docId, desc (3) flag, desc.
   *
   * @see org.apache.hadoop.io.WritableComparator#compare(byte[], int, int, byte[], int, int)
   */
  @Override
  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
    int c = compareNoBooleanCheck(b1, s1, l1, b2, s2, l2);
    if (c != 0) {
      // they have different termId/docId
      return c;
    }

    // advance start offsets
    s1 += 8;
    s2 += 8;

    // return reverse sorting of the boolean as a byte (1 == true, 0 == false)
    return -compare(b1[s1], b2[s2]);
  }
}
