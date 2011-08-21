package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.document.SimpleTextMatrixMapper;

import java.util.Map;

public class SimpleTextMatrixVectorizer extends AbstractJob {
  @Override public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.helpOption());
    addOption(DefaultOptionCreator.overwriteOption().create());
    // TODO: column normalization (idf)
    // TODO: row normalization (L1, L2)

    Map<String, String> args = parseArguments(strings);
    if(args == null) {
      return -1;
    }

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();

    Configuration conf = getConf();
    String jobName = "Simple text matrix vectorization";
    Job job = new Job(conf, jobName);
    job.setMapperClass(SimpleTextMatrixMapper.class);
    job.setReducerClass(Reducer.class);
    job.setJarByClass(SimpleTextMatrixVectorizer.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.addInputPath(job, inputPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    if(args.containsKey(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(conf, outputPath);
    }

    return job.waitForCompletion(true) ? 1 : 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SimpleTextMatrixVectorizer(), args);
  }
}
