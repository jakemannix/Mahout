package org.apache.mahout.clustering.lda.cvb;

import java.io.DataOutput;
import java.io.IOException;

public class StringDataOutput implements DataOutput {
  private StringBuffer buffer = new StringBuffer();

  public String toString() {
    return buffer.toString();
  }

  @Override public void write(int o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void write(byte[] bytes) throws IOException {
    throw new UnsupportedOperationException("Not supported");
  }

  @Override public void write(byte[] bytes, int offset, int len) throws IOException {
    throw new UnsupportedOperationException("Not supported");
  }

  @Override public void writeBoolean(boolean o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeByte(int o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeShort(int o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeChar(int o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeInt(int o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeLong(long o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeFloat(float o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeDouble(double o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeBytes(String o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeChars(String o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }

  @Override public void writeUTF(String o) throws IOException {
    buffer.append(String.valueOf(o)).append(" ");
  }
}
