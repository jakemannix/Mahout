/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils.vectors;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility which reads {@code SequenceFile<K, V>} where all keys are already
 * unique and creates two outputs: (1) {@code SequenceFile<IntWritable, V>}
 * which replaces the keys in the original input with integers (guaranteed to be
 * sequential and start at zero) and (2) a {@code SequenceFile<IntWritable, K>}
 * mapping from the new integer keys to the original input keys. This job will
 * fail if the input contains more unique keys (input pairs) than
 * {@link Integer#MAX_VALUE}.
 *
 * @author Andy Schlaikjer
 */
public class RowIdJob extends AbstractJob {
  /**
   * Writable which encodes path and start offset from a {@link FileSplit}.
   *
   * @author Andy Schlaikjer
   */
  public static final class FileSplitWritable implements
      WritableComparable<FileSplitWritable> {
    private final Text path;
    private final LongWritable start;

    public FileSplitWritable(Text path, LongWritable start) {
      super();
      this.path = path;
      this.start = start;
    }

    public FileSplitWritable() {
      this(new Text(), new LongWritable());
    }

    public FileSplitWritable(String path, long start) {
      this.path = new Text(path);
      this.start = new LongWritable(start);
    }

    public FileSplitWritable(FileSplitWritable other) {
      this(new Text(other.path), new LongWritable(other.start.get()));
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((path == null) ? 0 : path.hashCode());
      result = prime * result + ((start == null) ? 0 : start.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      FileSplitWritable other = (FileSplitWritable) obj;
      if (path == null) {
        if (other.path != null) return false;
      } else if (!path.equals(other.path)) return false;
      if (start == null) {
        if (other.start != null) return false;
      } else if (!start.equals(other.start)) return false;
      return true;
    }

    @Override
    public int compareTo(FileSplitWritable other) {
      if (this == other) return 0;
      if (other == null) return 1;
      if (path == null) {
        if (other.path != null) return -1;
      } else {
        int c = path.compareTo(other.path);
        if (c != 0) return c;
      }
      if (start == null) {
        if (other.start != null) return -1;
      } else {
        int c = start.compareTo(other.start);
        if (c != 0) return c;
      }
      return 0;
    }

    @Override
    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(path).append(':').append(start);
      return sb.toString();
    }

    @Override
    public void write(DataOutput out) throws IOException {
      path.write(out);
      start.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      path.readFields(in);
      start.readFields(in);
    }

    static {
      // we only care about exact binary equality of FileSplitWritable instances
      WritableComparator.define(FileSplitWritable.class,
          new BytesWritable.Comparator());
    }
  }

  /**
   * Reports the number of entries within each FileSplit.
   *
   * @author Andy Schlaikjer
   * @param <IK>
   * @param <IV>
   */
  public static class S1Mapper<IK,IV> extends
      Mapper<IK,IV,FileSplitWritable,IntWritable> {
    @Override
    public void run(Context context) throws IOException, InterruptedException {
      FileSplit fileSplit = (FileSplit) context.getInputSplit();
      FileSplitWritable key = new FileSplitWritable(new Text(fileSplit
          .getPath().toString()), new LongWritable(fileSplit.getStart()));
      long count = 0;
      while (context.nextKeyValue()) {
        count++;
        Preconditions.checkArgument(count <= Integer.MAX_VALUE,
            "Entry count exceeds Integer.MAX_VALUE");
      }
      context.write(key, new IntWritable((int) count));
    }
  }

  /**
   * Builds an in-memory {@code Map<FileSplitWritable, Integer>} from
   * {@code SequenceFile<FileSplitWritable, IntWritable>}s, and uses this to
   * transform input (K, V) pairs to output (IntWritable, V) and (IntWritable,
   * K) pairs.
   *
   * @author Andy Schlaikjer
   * @param <IK>
   * @param <IV>
   */
  public static class S2Mapper<IK,IV> extends Mapper<IK,IV,IntWritable,IV> {
    private final Map<FileSplitWritable,Pair<Integer,Integer>> splitIndices = new HashMap<FileSplitWritable,Pair<Integer,Integer>>();
    private Class<IK> inputKeyClass;

    @Override
    public void run(Context context) throws IOException, InterruptedException {
      initialize(context);

      // get index offset for current FileSplit
      FileSplit split = (FileSplit) context.getInputSplit();
      FileSplitWritable key = new FileSplitWritable(split.getPath().toString(),
          split.getStart());
      Pair<Integer,Integer> splitIndex = splitIndices.get(key);
      Preconditions.checkNotNull(splitIndex,
          "Integer offset for FileSplit '%s' is undefined", key);

      SequenceFile.Writer indexPartWriter = null;
      try {

        // open output file in which to record (IntWritable, IK) pairs
        // TODO verify the following side-effect path creation technique works
        // FileOutputFormat.getPathForWorkFile(context, "", "");
        Path indexPartPath = context.getWorkingDirectory().suffix(
            String.format("%s/part-m-%05d", OUTPUT_INDEX_SUBDIR, context
                .getTaskAttemptID().getId()));
        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);
        indexPartWriter = SequenceFile.createWriter(fs, conf, indexPartPath,
            IntWritable.class, inputKeyClass);

        // iterate over (IK, IV) pairs in current FileSplit
        int expectedCount = splitIndex.getFirst();
        int count = 0;
        int offset = splitIndex.getSecond();
        IntWritable index = new IntWritable(offset);
        while (context.nextKeyValue()) {
          // sanity check
          Preconditions.checkState(count <= expectedCount,
              "Input entry count exceeds expected count of %s", expectedCount);

          // write output pairs
          context.write(index, context.getCurrentValue());
          indexPartWriter.append(index, context.getCurrentKey());

          // update count and output index
          index.set(++count + offset);
        }

      } finally {
        if (indexPartWriter != null) indexPartWriter.close();
      }
    }

    private void initialize(Context context) throws IOException {
      if (!splitIndices.isEmpty()) return;

      // init input key class (used during index creation)
      Configuration conf = context.getConfiguration();
      inputKeyClass = forName(getRequiredParam(conf, INPUT_KEY_CLASS_ATTR));

      // init FileSplit infos
      loadFileSplitInfo(context);
    }

    private void loadFileSplitInfo(Context context) throws IOException {
      // grab path to input FileSplit infos
      Configuration conf = context.getConfiguration();
      String fileSplitInfoPath = getRequiredParam(conf,
          INPUT_FILESPLITINFO_PATH_ATTR);

      // load counts and index offsets into map
      // WARNING this assumes that all Mappers will see part-m-* files in same
      // order
      log.info("Loading FileSplit info from directory '" + fileSplitInfoPath
          + "'");
      long totalCount = 0;
      FileSystem fs = FileSystem.get(conf);
      for (FileStatus s : fs.globStatus(new Path(fileSplitInfoPath)
          .suffix("/part-m-*"))) {
        Path p = s.getPath();
        SequenceFileIterable<FileSplitWritable,IntWritable> itr = new SequenceFileIterable<FileSplitWritable,IntWritable>(
            p, true, conf);
        for (Pair<FileSplitWritable,IntWritable> pair : itr) {
          int count = pair.getSecond().get();

          // pair contains FileSplit (count, index offset)
          Pair<Integer,Integer> value = new Pair<Integer,Integer>(count,
              (int) totalCount);
          splitIndices.put(new FileSplitWritable(pair.getFirst()), value);

          // update total count and test for int overflow
          totalCount += count;
          Preconditions.checkArgument(totalCount <= Integer.MAX_VALUE,
              "Total entry count %s is greater than Integer.MAX_VALUE",
              totalCount);
        }
      }
      log.info(
          "Loaded FileSplit info for {} splits containing {} entries total",
          splitIndices.size(), totalCount);
    }

    @SuppressWarnings("unchecked")
    private <T> Class<T> forName(String className) throws IOException {
      try {
        return (Class<T>) Class.forName(className);
      } catch (ClassNotFoundException e) {
        throw new IOException(e);
      }
    }
  }

  private static String getRequiredParam(Configuration conf, String name) {
    String value = conf.get(name);
    Preconditions.checkNotNull(value, "Required parameter '%s' is undefined",
        name);
    return value;
  }

  private static final Logger log = LoggerFactory.getLogger(RowIdJob.class);
  public static final String OUTPUT_FILESPLITINFO_SUBDIR = "fileSplitInfo";
  public static final String OUTPUT_INDEX_SUBDIR = "docIndex";
  public static final String OUTPUT_MATRIX_SUBDIR = "matrix";

  private static final String INPUT_FILESPLITINFO_PATH_ATTR = RowIdJob.class
      .getName() + ".input.fileSplitInfo.path";
  private static final String INPUT_KEY_CLASS_ATTR = RowIdJob.class.getName()
      + ".input.key.class";

  @Override
  public int run(String[] args) throws Exception {
    // define options and parse arguments
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption("inputKeyClass", "ik",
        "Name of key class within input SequenceFile data", true);
    addOption("inputValueClass", "iv",
        "Name of value class within input SequenceFile data", true);
    Map<String,String> options = parseArguments(args);

    Class<?> inputKeyClass = Class.forName(options.get("ik"));
    Class<?> inputValueClass = Class.forName(options.get("iv"));

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path stage1OutputPath = outputPath
        .suffix("/" + OUTPUT_FILESPLITINFO_SUBDIR);
    Path stage2IndexOutputPath = outputPath.suffix("/" + OUTPUT_INDEX_SUBDIR);
    Path stage2MatrixOutputPath = outputPath.suffix("/" + OUTPUT_MATRIX_SUBDIR);

    runStage1(inputPath, stage1OutputPath);
    runStage2(inputPath, inputKeyClass, inputValueClass, stage1OutputPath,
        stage2IndexOutputPath, stage2MatrixOutputPath);
    return 0;
  }

  private void runStage1(Path inputPath, Path outputPath) throws Exception {
    Job job = new Job(getConf(), RowIdJob.class.getSimpleName()
        + " Stage 1: Generating FileSplit info");
    job.setJarByClass(getClass());

    job.setMapperClass(S1Mapper.class);
    job.setOutputKeyClass(FileSplitWritable.class);
    job.setOutputValueClass(IntWritable.class);
    job.setNumReduceTasks(0);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    SequenceFileInputFormat.addInputPath(job, inputPath);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    SequenceFileOutputFormat.setOutputPath(job, outputPath);

    if (!job.waitForCompletion(true)) throw new IllegalStateException(
        "Failed to generate FileSplit info");
  }

  private void runStage2(Path inputPath, Class<?> inputKeyClass,
      Class<?> inputValueClass, Path fileSplitInfoInputPath,
      Path indexOutputPath, Path matrixOutputPath) throws Exception {
    Job job = new Job(getConf(), RowIdJob.class.getSimpleName()
        + " Stage 2: Transforming (K, V) to {(int, V), (int, K)}");
    job.setJarByClass(getClass());

    Configuration conf = job.getConfiguration();
    conf.set(INPUT_KEY_CLASS_ATTR, inputKeyClass.getName());
    conf.set(INPUT_FILESPLITINFO_PATH_ATTR, fileSplitInfoInputPath.toString());

    job.setMapperClass(S2Mapper.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(inputValueClass);
    job.setNumReduceTasks(0);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    SequenceFileInputFormat.addInputPath(job, inputPath);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    SequenceFileOutputFormat.setOutputPath(job, matrixOutputPath);
    if (!job.waitForCompletion(true)) throw new IllegalStateException(
        "Failed to generate FileSplit info");
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowIdJob(), args);
  }
}
