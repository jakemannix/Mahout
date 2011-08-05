package org.apache.mahout.clustering.lda.cvb;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class TestCVBComparators extends MahoutTestCase {

  @Test
  public void testCVBSortingComparator() throws Exception {
    CVBKey key1 = new CVBKey();
    key1.setTermId(1);
    key1.setDocId(101);
    key1.setB(false);
    CVBKey key2 = new CVBKey();
    key2.setTermId(1);
    key2.setDocId(101);
    key2.setB(true);
    compareBothWays(key1, key2, 0, 1);
    key1.setB(true);
    compareBothWays(key1, key2, 0, 0);
    key1.setTermId(2);
    compareBothWays(key1, key2, -1, -1);
    key2.setDocId(102);
    compareBothWays(key1, key2, -1, -1);
    key1.setTermId(1);
    compareBothWays(key1, key2, 1, 1);
  }

  public static void compareBothWays(CVBKey key1, CVBKey key2, int expectedNoBoolean, int expected)
      throws IOException {
    ByteArrayOutputStream buffer1 = new ByteArrayOutputStream();
    DataOutputStream outputStream1 = new DataOutputStream(buffer1);
    key1.write(outputStream1);
    byte[] key1Bytes = buffer1.toByteArray();

    ByteArrayOutputStream buffer2 = new ByteArrayOutputStream();
    DataOutputStream outputStream2 = new DataOutputStream(buffer2);
    key2.write(outputStream2);
    byte[] key2Bytes = buffer2.toByteArray();

    int result = CVBSortingComparator.compareNoBooleanCheck(key1Bytes, 0, key1Bytes.length,
        key2Bytes, 0, key2Bytes.length);
    assertEquals(expectedNoBoolean, result);

    result = new CVBSortingComparator().compare(key1Bytes, 0, key1Bytes.length,
        key2Bytes, 0, key2Bytes.length);
    assertEquals(expected, result);
  }

}
