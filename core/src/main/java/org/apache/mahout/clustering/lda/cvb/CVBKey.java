package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * serialized format: bytes:
 * [_ _ _ _][_ _ _ _][_][_ _ _ _]
 *   termId    docId  b  branch
 */
public class CVBKey implements WritableComparable<CVBKey> {
  private int termId;
  private int docId;
  private boolean b;
  private AggregationBranch branch;

  public CVBKey() {
    initialize();
  }

  public CVBKey(CVBKey other) {
    termId = other.getTermId();
    docId = other.getDocId();
    b = other.isB();
    branch = other.getBranch();
  }

  private void initialize() {
    termId = -1;
    docId = -1;
    b = false;
    branch = null;
  }

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
    if(this.branch != cvbKey.branch && (termId+1)*(docId+1) == 0) {
      throw new IllegalStateException("docId, termId equal, but different branches:" +
        toString() + " : " + cvbKey.toString());
    }
    return 0;
  }

  @Override public void write(DataOutput out) throws IOException {
    out.writeInt(termId);
    out.writeInt(docId);
    out.writeBoolean(b);
    out.writeInt(branch == null ? 0 : branch.ordinal() + 1);
    checkState(out);
  }

  private void checkState(DataOutput out) {
    if(termId == -1 || docId == -1) {
      if(branch != AggregationBranch.of(termId, docId)
         && (out == null || !(out instanceof StringDataOutput))) {
        throw new IllegalStateException("Should not be writing: " + toString());
      }
    }
  }

  @Override public void readFields(DataInput in) throws IOException {
    initialize();
    termId = in.readInt();
    docId = in.readInt();
    b = in.readBoolean();
    int i = in.readInt();
    branch = i == 0 ? null : AggregationBranch.values()[i - 1];
    checkState(null);
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

  public String toString() {
    DataOutput output = new StringDataOutput();
    try {
      write(output);
      return output.toString();
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) { return true; }
    if (o == null || getClass() != o.getClass()) { return false; }

    CVBKey cvbKey = (CVBKey) o;

    if (b != cvbKey.b) { return false; }
    if (docId != cvbKey.docId) { return false; }
    if (termId != cvbKey.termId) { return false; }

    return true;
  }

  @Override
  public int hashCode() {
    int result = termId;
    result = 31 * result + docId;
    result = 31 * result + (b ? 1 : 0);
    return result;
  }

  public AggregationBranch getBranch() {
    return branch;
  }

  public void setBranch(AggregationBranch branch) {
    this.branch = branch;
  }
}
