package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class CVBKey implements WritableComparable<CVBKey> {
  private int termId;
  private int docId;
  private boolean b;

  @Override public int compareTo(CVBKey cvbKey) {
    int i = compareToOnlyIgnoreBoolean(cvbKey);
    if(i != 0) {
      return i;
    }
    if(this.b && !cvbKey.b) {
      return 1;
    }
    if(!this.b && cvbKey.b) {
      return -1;
    }
    return 0;
  }

  public int compareToOnlyIgnoreBoolean(CVBKey cvbKey) {
    if(this.termId > cvbKey.termId) {
      return 1;
    }
    if(this.termId < cvbKey.termId) {
      return -1;
    }
    if(this.docId > cvbKey.docId) {
      return 1;
    }
    if(this.docId < cvbKey.docId) {
      return -1;
    }
    return 0;
  }

  @Override public void write(DataOutput out) throws IOException {
    out.writeInt(termId);
    out.writeInt(docId);
    out.writeBoolean(b);
  }

  @Override public void readFields(DataInput in) throws IOException {
    termId = in.readInt();
    docId = in.readInt();
    b = in.readBoolean();
  }

  public int getTermId() {
    return termId;
  }

  public void setTermId(int termId) {
    this.termId = termId;
  }

  public int getDocId() {
    return docId;
  }

  public void setDocId(int docId) {
    this.docId = docId;
  }

  public boolean isB() {
    return b;
  }

  public void setB(boolean b) {
    this.b = b;
  }
}
