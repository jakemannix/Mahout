package org.apache.mahout.vectorizer.document;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.CharTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;

public class SimpleTextMatrixMapper extends Mapper<LongWritable, Text, IntWritable, VectorWritable> {

  private VectorWritable vectorWritable = new VectorWritable();
  private IntWritable rowId = new IntWritable();

  @Override
  public void map(LongWritable offset, Text textLine, Context context)
      throws IOException, InterruptedException {
    String line = textLine.toString();
    int row = -1;
    Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, line.length() / 16);
    IntegerTokenizer tokenizer = new IntegerTokenizer(Version.LUCENE_31,
        new StringReader(line));
    CharTermAttribute termAtt = tokenizer.addAttribute(CharTermAttribute.class);
    while (tokenizer.incrementToken()) {
      if (termAtt.length() > 0) {
        int i = Integer.parseInt(new String(termAtt.buffer(), 0, termAtt.length()));
        if(row < 0) {
          row = i;
        } else {
          vector.set(i, 1);
        }
      }
    }
    rowId.set(row);
    vectorWritable.set(vector);
    context.write(rowId, vectorWritable);
  }

  private static class IntegerTokenizer extends CharTokenizer {
    public IntegerTokenizer(Version matchVersion, Reader input) {
      super(matchVersion, input);
    }
    @Override protected boolean isTokenChar(int c) {
      return Character.isDigit(c);
    }
  }
}
