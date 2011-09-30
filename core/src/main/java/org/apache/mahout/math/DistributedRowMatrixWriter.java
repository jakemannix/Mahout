package org.apache.mahout.math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;

import java.io.IOException;

public class DistributedRowMatrixWriter {

  public static void write(Path outputDir, Configuration conf, VectorIterable matrix)
      throws IOException {
    FileSystem fs = outputDir.getFileSystem(conf);
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outputDir,
        IntWritable.class, VectorWritable.class);
    IntWritable topic = new IntWritable();
    VectorWritable vector = new VectorWritable();
    for(MatrixSlice slice : matrix) {
      topic.set(slice.index());
      vector.set(slice.vector());
      writer.append(topic, vector);
    }
    writer.close();

  }

}
