/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils.vectors;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Iterator;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Can read in a {@link SequenceFile} of {@link Vector}s and dump
 * out the results using {@link Vector#asFormatString()} to either the console or to a
 * file.
 */
public final class VectorDumper {

  private static final Logger log = LoggerFactory.getLogger(VectorDumper.class);

  private VectorDumper() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
        abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Sequence File containing the Vectors").withShortName("s").create();
    Option vectorAsKeyOpt = obuilder.withLongName("useKey").withRequired(false).withDescription(
        "If the Key is a vector, then dump that instead").withShortName("u").create();
    Option printKeyOpt =
        obuilder.withLongName("printKey").withRequired(false).withDescription(
            "Print out the key as well, delimited by a tab (or the value if useKey is true)")
            .withShortName("p")
            .create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
        abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription(
        "The dictionary file. ").withShortName("d").create();
    Option dictTypeOpt =
        obuilder.withLongName("dictionaryType").withRequired(false).withArgument(
            abuilder.withName("dictionaryType").withMinimum(1).withMaximum(1).create())
            .withDescription(
                "The dictionary file type (text|sequencefile)").withShortName("dt").create();
    Option csvOpt = obuilder.withLongName("csv").withRequired(false).withDescription(
        "Output the Vector as CSV.  Otherwise it substitutes in the terms for vector cell entries")
        .withShortName("c").create();
    Option namesAsCommentsOpt =
        obuilder
            .withLongName("namesAsComments")
            .withRequired(false)
            .withDescription(
                "If using CSV output, optionally add a comment line for each NamedVector (if the vector is one) printing out the name")
            .withShortName("n").create();
    Option sortVectorsOpt =
        obuilder.withLongName("sortVectors").withRequired(false).withDescription(
            "Sort output key/value pairs of the vector entries in abs magnitude descending order")
            .withShortName("sort").create();
    Option sizeOpt = obuilder.withLongName("sizeOnly").withRequired(false).
        withDescription("Dump only the size of the vector").withShortName("sz").create();
    Option numItemsOpt = obuilder.withLongName("numItems").withShortName("n").withRequired(false).withArgument(
        abuilder.withName("numItems").withMinimum(1).withMaximum(1).create()).
        withDescription("Output at most <numItems> key-value pairs").create();
    Option helpOpt =
        obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
            .create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt).withOption(
        dictTypeOpt).withOption(dictOpt).withOption(csvOpt).withOption(vectorAsKeyOpt).withOption(
        printKeyOpt).withOption(sortVectorsOpt).withOption(helpOpt).withOption(numItemsOpt).withOption(sizeOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        printHelp(group);
        return;
      }

      if (cmdLine.hasOption(seqOpt)) {
        Configuration conf = new Configuration();
        Path pathPattern = new Path(cmdLine.getValue(seqOpt).toString());
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] inputPaths = fs.globStatus(pathPattern);

        boolean useCSV = cmdLine.hasOption(csvOpt);
        boolean sizeOnly = cmdLine.hasOption(sizeOpt);
        boolean namesAsComments = cmdLine.hasOption(namesAsCommentsOpt);
        boolean transposeKeyValue = cmdLine.hasOption(vectorAsKeyOpt);
        boolean printKey = cmdLine.hasOption(printKeyOpt);
        boolean sortVectors = cmdLine.hasOption(sortVectorsOpt);

        String[] dictionary = null;
        if (cmdLine.hasOption(dictOpt)) {
          String dictFile = cmdLine.getValue(dictOpt).toString();
          String dictionaryType =
              cmdLine.hasOption(dictTypeOpt) ? cmdLine.getValue(dictTypeOpt).toString() : "text";
          log.info("Loading " + dictionaryType + " dictionary '" + dictFile + "'");
          if ("text".equals(dictionaryType)) {
            dictionary =
                VectorHelper.loadTermDictionary(new File(dictFile));
          } else if ("sequencefile".equals(dictionaryType)) {
            dictionary =
                VectorHelper.loadTermDictionary(conf, dictFile);
          } else {
            throw new OptionException(dictTypeOpt);
          }
        }

        Writer writer;
        if (cmdLine.hasOption(outputOpt)) {
          writer =
              Files.newWriter(new File(cmdLine.getValue(outputOpt).toString()), Charsets.UTF_8);
        } else {
          writer = new OutputStreamWriter(System.out);
        }

        try {
          if (useCSV && dictionary != null) {
            writer.write("#");
            for (int j = 0; j < dictionary.length; j++) {
              writer.write(dictionary[j]);
              if (j < dictionary.length - 1) {
                writer.write(',');
              }
            }
            writer.write('\n');
          }

          long numItems = Long.MAX_VALUE;
          if (cmdLine.hasOption(numItemsOpt)) {
            numItems = Long.parseLong(cmdLine.getValue(numItemsOpt).toString());
            writer.append("#Max Items to dump: ").append(String.valueOf(numItems)).append('\n');
          }

          int fileIndex = 0;
          for (FileStatus s : inputPaths) {
            Path path = s.getPath();
            log.info("Processing '" + path + "' (" + (++fileIndex) + "/" + inputPaths.length + ")");
            dump(conf, s.getPath(), writer, dictionary, sortVectors, numItems, printKey,
                transposeKeyValue, sizeOnly, useCSV, namesAsComments);
          }

        } finally {
          Closeables.closeQuietly(writer);
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      printHelp(group);
    }
  }

  private static void dump(Configuration conf, Path path, Writer writer,
      String[] dictionary, boolean sortVectors, long numItems, boolean printKey,
      boolean transposeKeyValue, boolean sizeOnly, boolean useCSV, boolean namesAsComments)
      throws IOException {
    SequenceFileIterable<Writable, Writable> iterable =
        new SequenceFileIterable<Writable, Writable>(path, true, conf);
    Iterator<Pair<Writable, Writable>> iterator = iterable.iterator();
    long i = 0;
    long count = 0;
    while (iterator.hasNext() && count < numItems) {
      Pair<Writable, Writable> record = iterator.next();
      Writable keyWritable = record.getFirst();
      Writable valueWritable = record.getSecond();
      if (printKey) {
        Writable notTheVectorWritable = transposeKeyValue ? valueWritable : keyWritable;
        writer.write(notTheVectorWritable.toString());
        writer.write('\t');
      }
      VectorWritable vectorWritable =
          (VectorWritable) (transposeKeyValue ? keyWritable : valueWritable);
      Vector vector = vectorWritable.get();
      if (sizeOnly) {
        if (vector instanceof NamedVector) {
          writer.write(((NamedVector) vector).getName());
          writer.write(":");
        } else {
          writer.write(String.valueOf(i++));
          writer.write(":");
        }
        writer.write(String.valueOf(vector.size()));
        writer.write('\n');
      } else {
        String fmtStr;
        if (useCSV) {
          fmtStr = VectorHelper.vectorToCSVString(vector, namesAsComments);
        } else {
          fmtStr = sortVectors ? VectorHelper.vectorToSortedString(vector, dictionary)
              : vector.asFormatString(dictionary);
        }
        writer.write(fmtStr);
        writer.write('\n');
      }
      count++;
    }
  }

  private static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
}
