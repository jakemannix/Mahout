package org.apache.mahout.vectorizer.document;

import com.google.common.base.Charsets;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

public class TextFormatTokenizerMapper extends AbstractTokenizerMapper<LongWritable, Text> {

  public static final String SEPARATOR = TextFormatTokenizerMapper.class.getName() + ".separator";

  private String separator = null;

  private int separatorIndex(Text line) {
    int i = -1;
    if(separator != null) {
      i = line.find(separator);
    } else {
      while(Character.isDigit(line.charAt(++i)));
    }
    if(i < 0) {
      throw new IllegalArgumentException("Input lines must contain separator: [" + separator + "]");
    }
    return i;
  }

  @Override
  public String extractDocumentId(LongWritable offset, Text line) {
    int i = separatorIndex(line);
    byte[] bytes = line.getBytes();
    return new String(bytes, 0, i, Charsets.UTF_8);
  }

  @Override
  public String extractDocument(LongWritable offset, Text line) {
    int i = separatorIndex(line);
    byte[] bytes = line.getBytes();
    return new String(bytes, i, bytes.length - i, Charsets.UTF_8);
  }

  @Override
  public void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    separator = context.getConfiguration().get(SEPARATOR);
  }
}
