package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.EnumMap;

public class CVBTuple implements Writable {
  
  // a, i, C_ai, [T_ax], [T_x], [D_ix]
  // g_aix = (T_ax + eta)/(T_x + eta*W) * (D_ix + alpha)

  private int termId;
  private int documentId;
  private double itemCount;
  private EnumMap<AggregationBranch, Byte> branchMasks
      = Maps.newEnumMap(ImmutableMap.of(AggregationBranch.TOPIC_TERM, (byte)0x2,
                                        AggregationBranch.DOC_TOPIC, (byte)0x4,
                                        AggregationBranch.TOPIC_SUM, (byte)0x8));
  private EnumMap<AggregationBranch, double[]> counts
      = new EnumMap<AggregationBranch, double[]>(AggregationBranch.class);

  public void accumulate(AggregationBranch branch, CVBTuple other) {
    double[] currentCounts = counts.get(branch);
    double[] otherCount = other.getCount(branch);
    for(int x=0; x<currentCounts.length; x++) {
      currentCounts[x] += otherCount[x];
    }
  }

  @Override public void write(DataOutput out) throws IOException {
    out.writeByte(existenceByte());
    out.writeInt(termId);
    out.writeInt(documentId);
    if(itemCount > 0) {
      out.writeDouble(itemCount);
    }
    for(AggregationBranch branch : AggregationBranch.values()) {
      writeArray(out, counts.get(branch));
    }
  }

  private void writeArray(DataOutput out, double[] a) throws IOException {
    if(a != null) {
      out.writeInt(a.length);
      for(double d : a) {
        out.writeDouble(d);
      }
    }
  }

  private double[] readArray(DataInput in) throws IOException {
    int length = in.readInt();
    double[] a = new double[length];
    for(int i = 0; i < length; i++) {
      a[i] = in.readDouble();
    }
    return a;
  }


  @Override public void readFields(DataInput in) throws IOException {
    byte existenceByte = in.readByte();
    termId = in.readInt();
    documentId = in.readInt();
    if((existenceByte & 0x1) != 0) {
      itemCount = in.readDouble();
    }
    for(AggregationBranch branch : AggregationBranch.values()) {
      if((existenceByte & branchMasks.get(branch)) != 0) {
        double[] a = readArray(in);
        counts.put(branch, a);
      }
    }
  }

  private byte existenceByte() {
    byte b = (byte)(itemCount > 0 ? 0x1 : 0);
    for(AggregationBranch branch : AggregationBranch.values()) {
      if(counts.containsKey(branch)) {
        b = (byte)(b | branchMasks.get(branch));
      }
    }
    return b;
  }

  public double getItemCount() {
    return itemCount;
  }

  public void setItemCount(double itemCount) {
    this.itemCount = itemCount;
  }

  public void setCount(AggregationBranch branch, double[] count) {
    counts.put(branch, count);
  }

  public double[] getCount(AggregationBranch branch) {
    return counts.get(branch);
  }

  public int getDocumentId() {
    return documentId;
  }

  public void setDocumentId(int documentId) {
    this.documentId = documentId;
  }

  public int getTermId() {
    return termId;
  }

  public void setTermId(int termId) {
    this.termId = termId;
  }

}
