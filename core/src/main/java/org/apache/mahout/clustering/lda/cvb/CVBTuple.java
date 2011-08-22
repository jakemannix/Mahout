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

  private int termId = -1;
  private int documentId = -1;
  private double itemCount = -1;
  private static EnumMap<AggregationBranch, Byte> branchMasks
      = Maps.newEnumMap(ImmutableMap.of(AggregationBranch.TOPIC_TERM, (byte)0x2,
                                        AggregationBranch.DOC_TOPIC, (byte)0x4,
                                        AggregationBranch.TOPIC_SUM, (byte)0x8));
  private EnumMap<AggregationBranch, double[]> counts
      = new EnumMap<AggregationBranch, double[]>(AggregationBranch.class);

  private int topic = -1;
  private double count = -1;

  public CVBTuple() {
    initialize();
  }

  public CVBTuple(CVBTuple other) {
    termId = other.termId;
    documentId = other.documentId;
    itemCount = other.itemCount;
    for(AggregationBranch branch : AggregationBranch.values()) {
      if(other.hasData(branch)) {
        counts.put(branch, other.getCount(branch).clone());
      }
    }
    topic = other.topic;
    count = other.count;
  }

  private void initialize() {
    termId = -1;
    documentId = -1;
    itemCount = -1;
    topic = -1;
    count = -1;
    counts.clear();
  }

  public void accumulate(AggregationBranch branch, CVBTuple other) {
    double[] otherCount = other.getCount(branch);
    double[] currentCounts = counts.get(branch);
    if(currentCounts == null) {
      currentCounts = new double[otherCount.length];
      counts.put(branch, currentCounts);
    }
    for(int x=0; x<currentCounts.length; x++) {
      currentCounts[x] += otherCount[x];
    }
  }

  public void sparseAccumulate(AggregationBranch branch, CVBTuple other) {
    counts.get(branch)[other.topic] += other.count;
  }

  @Override public void write(DataOutput out) throws IOException {
    out.writeByte(existenceByte());
    if(itemCount > 0) {
      out.writeInt(termId);
      out.writeInt(documentId);
      out.writeDouble(itemCount);
    }
    for(AggregationBranch branch : AggregationBranch.values()) {
      writeArray(out, counts.get(branch));
    }
    if(topic >= 0 && count >= 0) {
      out.writeInt(topic);
      out.writeDouble(count);
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

  @Override
  public boolean equals(Object o) {
    if (this == o) { return true; }
    if (o == null || getClass() != o.getClass()) { return false; }

    CVBTuple tuple = (CVBTuple) o;

    if (documentId != tuple.documentId) { return false; }
    if (Double.compare(tuple.itemCount, itemCount) != 0) { return false; }
    if (termId != tuple.termId) { return false; }
    if(!counts.keySet().equals(tuple.counts.keySet())) { return false; }
    for (AggregationBranch branch : counts.keySet()) {
      double[] d = counts.get(branch);
      double[] d1 = tuple.getCount(branch);
      for(int i=0; i<d.length; i++) {
        if(Double.compare(d[i], d1[i]) != 0) {
          return false;
        }
      }
    }

    return true;
  }

  @Override
  public int hashCode() {
    int result;
    long temp;
    result = termId;
    result = 31 * result + documentId;
    temp = itemCount != +0.0d ? Double.doubleToLongBits(itemCount) : 0L;
    result = 31 * result + (int) (temp ^ (temp >>> 32));
    int countHash = 0;
    for (AggregationBranch branch : counts.keySet()) {
      double[] d = counts.get(branch);
      if(d == null) continue;
      for(int i=0; i<d.length; i++) {
        temp = d[i] != +0.0d ? Double.doubleToLongBits(d[i]) : 0L;
        countHash += (31 * countHash + (int)(temp ^ (temp >>> 32)));
      }
    }
    result = 31 * result + countHash;
    return result;
  }

  @Override public void readFields(DataInput in) throws IOException {
    initialize();
    byte existenceByte = in.readByte();
    if((existenceByte & 0x1) != 0) {
      termId = in.readInt();
      documentId = in.readInt();
      itemCount = in.readDouble();
    }
    for(AggregationBranch branch : AggregationBranch.values()) {
      if((existenceByte & branchMasks.get(branch)) != 0) {
        double[] a = readArray(in);
        counts.put(branch, a);
      }
    }
    if((existenceByte & 0x16) != 0) {
      topic = in.readInt();
      count = in.readDouble();
    }
  }

  public boolean hasAnyData() {
    for(AggregationBranch branch : AggregationBranch.values()) {
      if(hasData(branch)) {
        return true;
      }
    }
    return false;
  }

  public boolean hasAllData() {
    for(AggregationBranch branch : AggregationBranch.values()) {
      if(!hasData(branch)) {
        return false;
      }
    }
    return true;
  }

  public boolean hasData(AggregationBranch branch) {
    return counts.get(branch) != null;
  }

  private byte existenceByte() {
    byte b = (byte)(itemCount > 0 ? 0x1 : 0);
    for(AggregationBranch branch : AggregationBranch.values()) {
      if(counts.containsKey(branch)) {
        b = (byte)(b | branchMasks.get(branch));
      }
    }
    if(topic >= 0 && count >= 0) {
      b = (byte)(b | 0x16);
    }
    return b;
  }

  public double getItemCount() {
    return itemCount;
  }

  public void setItemCount(double itemCount) {
    this.itemCount = itemCount;
  }

  public int getTopic() {
    return topic;
  }

  public double getCount() {
    return count;
  }

  public void setTopic(int topic) {
    this.topic = topic;
  }

  public void setCount(double count) {
    this.count = count;
  }

  public void clearCounts() {
    counts.clear();
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

  public String toString() {
    DataOutput output = new StringDataOutput();
    try {
      write(output);
      return output.toString();
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
