package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparator;

/**
 * serialized format: bytes:
 * [_ _ _ _][_ _ _ _][_][_ _ _ _]
 *   termId    docId  b  branch
 */
public class CVBSortingComparator extends WritableComparator {
  /**
   * Sorts raw CVBKey bytes by (1) termId, desc (2) docId, desc.
   */
  public static int compareNoBooleanCheck(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
    // compare termIds
    int c = -(readInt(b1, s1) - readInt(b2, s2));
    if (c != 0) {
      return c;
    }

    // advance start offsets
    s1 += 4;
    s2 += 4;

    // compare docIds
    return -(readInt(b1, s1) - readInt(b2, s2));
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
    return -(b1[s1] - b2[s2]);
  }
}
