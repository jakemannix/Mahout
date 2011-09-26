package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.common.DummyRecordWriter;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


public class TestCVBStage1 extends MahoutTestCase {

  public static double[][] corpus = { {1, 1, 1, 0, 0, 0}, {1, 0, 0, 1, 1, 0}, {1, 0, 0, 0, 0, 1},
                               {0, 4, 3, 0, 0, 1}, {1, 1, 0, 3, 4, 1}, {0, 1, 0, 2, 0, 1},
                               {0, 0, 1, 5, 2, 1}, {2, 3, 0, 0, 1, 0}, {0, 0, 0, 3, 0, 3},
                               {0, 1, 3, 1, 0, 0}, {1, 0, 1, 2, 1, 0}, {0, 3, 2, 0, 1, 1} };


  private FileSystem fs;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    conf.set(CVB0Mapper.NUM_TOPICS, String.valueOf(2));
    conf.set(CVB0Mapper.NUM_TERMS, String.valueOf(6));
    conf.set(CVB0Mapper.ALPHA, String.valueOf(0.01));
    conf.set(CVB0Mapper.ETA, String.valueOf(0.01));
    conf.set(CVB0Mapper.RANDOM_SEED, String.valueOf(1234));
    fs = FileSystem.get(conf);
    ClusteringTestUtils.writePointsToFile(points(), true, corpusPath(), fs, conf);
  }

  public void testStage0() throws Exception {
    CVB0Driver driver = new CVB0Driver();
    driver.runStage0(fs.getConf(), corpusPath(), getTopicModelStatePath());
    Path partFile = null;
    for(FileStatus status : fs.listStatus(CVB0Driver.stage1InputPath(getTopicModelStatePath(), 0),
        PathFilters.partFilter())) {
      partFile = status.getPath();
      break;
    }

    SequenceFile.Reader stateReader = new SequenceFile.Reader(fs, partFile, fs.getConf());
    CVBKey key = new CVBKey();
    CVBTuple tuple = new CVBTuple();
    int nonZeroCount = 0;
    while(stateReader.next(key, tuple)) {
      int termId = key.getTermId();
      int docId = key.getDocId();
      double count = tuple.getItemCount();
      assertEquals(corpus(docId, termId), count, 1e-9);
      nonZeroCount++;
      for(AggregationBranch branch : AggregationBranch.values()) {
        assertNull(tuple.getCount(branch));
      }
    }
    assertEquals(numNonZero(), nonZeroCount);
  }

 // @Test
  public void testShuffle() throws Exception {
    testStage0();
    Configuration configuration = fs.getConf();
    CVB0Mapper mapper = new CVB0Mapper();
    DummyRecordWriter<CVBKey, CVBTuple> writer = new DummyRecordWriter<CVBKey, CVBTuple>() {
      @Override
      public void write(CVBKey key, CVBTuple tuple) {
        super.write(new CVBKey(key), new CVBTuple(tuple));
      }
    };
    CVB0Mapper.Context context = DummyRecordWriter.build(mapper, configuration, writer);
    mapper.setup(context);

    Path partFile = null;
    for(FileStatus status : fs.listStatus(CVB0Driver.stage1InputPath(getTopicModelStatePath(), 0),
        PathFilters.partFilter())) {
      partFile = status.getPath();
      break;
    }

    SequenceFile.Reader stateReader = new SequenceFile.Reader(fs, partFile, fs.getConf());
    CVBKey key = new CVBKey();
    CVBTuple tuple = new CVBTuple();
    int nonZeroCount = 0;
    // let's mock map!
    while(stateReader.next(key, tuple)) {
      mapper.map(key, tuple, context);
    }
    List<Pair<CVBKey, CVBTuple>> mapOutput = Lists.newArrayList();
    for(Map.Entry<CVBKey, List<CVBTuple>> entry : writer.getData().entrySet()) {
      CVBKey outKey = entry.getKey();
      for(CVBTuple outTuple : entry.getValue()) {
        mapOutput.add(new Pair<CVBKey, CVBTuple>(outKey, outTuple));
      }
    }
    final CVBSortingComparator comparator = new CVBSortingComparator();
    Collections.sort(mapOutput, new Comparator<Pair<CVBKey, CVBTuple>>() {
      @Override public int compare(Pair<CVBKey, CVBTuple> a,
          Pair<CVBKey, CVBTuple> b) {
        return - comparator.compare(a.getFirst(), b.getFirst());
      }
    });
    Set<Pair<Integer,Integer>> expectedEntries = nonZeroCorpusEntries();
    Map<Pair<Integer,Integer>, List<Pair<AggregationBranch, double[]>>> partialSums = Maps.newHashMap();
    Map<Pair<Integer,Integer>, EnumMap<AggregationBranch,Integer>> entriesToTag = Maps.newHashMap();
    CVB0Reducer reducer = new CVB0Reducer();
    DummyRecordWriter<CVBKey, CVBTuple> reduceOutputWriter = new DummyRecordWriter<CVBKey, CVBTuple>();
    CVB0Reducer.Context reduceContext = DummyRecordWriter.build(reducer, configuration, reduceOutputWriter,
        CVBKey.class, CVBTuple.class);
    List<CVBTuple> tmp = Lists.newArrayList();
    final CVB0GroupingComparator groupingComparator = new CVB0GroupingComparator();
    Pair<CVBKey, CVBTuple> prev = null;
    CVBKey currentGroupingKey = null;
    for(Pair<CVBKey, CVBTuple> pair : mapOutput) {
      if(currentGroupingKey == null) {
        currentGroupingKey = new CVBKey(pair.getFirst());
      }
      if(prev == null) {
        prev = new Pair<CVBKey, CVBTuple>(new CVBKey(pair.getFirst()), new CVBTuple(pair.getSecond()));
        tmp.add(pair.getSecond());
      } else {
        if(pair.getFirst().compareToOnlyIgnoreBoolean(prev.getFirst()) == 0) {
          tmp.add(pair.getSecond());
          prev = pair;
        } else if(pair.getFirst().compareToOnlyIgnoreBoolean(prev.getFirst()) > 0) {
          fail("sort reversed!");
        } else {
          // reduce:
          for(CVBTuple tu : tmp) {
            Pair<Integer,Integer> p = Pair.of(currentGroupingKey.getDocId(),
                currentGroupingKey.getTermId());
            if(!partialSums.containsKey(p)) {
              partialSums.put(p,
                  Lists.<Pair<AggregationBranch,double[]>>newArrayList());
            }
            boolean onlyTag = true;
            for(AggregationBranch b : AggregationBranch.values()) {
              if(tu.hasData(b)) {
                partialSums.get(p).add(Pair.of(b, tu.getCount(b)));
                onlyTag = false;
              }
            }
            if(onlyTag) {
              Pair<Integer, Integer> pair1 = Pair.of(tu.getDocumentId(), tu.getTermId());
              if(!entriesToTag.containsKey(pair1)) {
                entriesToTag.put(pair1, Maps.<AggregationBranch, Integer>newEnumMap(AggregationBranch.class));
              }
              AggregationBranch b1 = AggregationBranch.of(currentGroupingKey.getTermId(),
                  currentGroupingKey.getDocId());
              if(!entriesToTag.get(pair1).containsKey(b1)) {
                entriesToTag.get(pair1).put(b1, 0);
              }
              entriesToTag.get(pair1).put(b1, entriesToTag.get(pair1).get(b1) + 1);
            }
          }
          reducer.reduce(currentGroupingKey, tmp, reduceContext);
          // moved on to next one.
          tmp.clear();
          prev = pair;
          currentGroupingKey = pair.getFirst();
          tmp.add(pair.getSecond());
        }
      }
    }
    // reduce last group
    if(!tmp.isEmpty()) {
      for(CVBTuple tu : tmp) {
        Pair<Integer,Integer> p = Pair.of(currentGroupingKey.getDocId(),
            currentGroupingKey.getTermId());
        if(!partialSums.containsKey(p)) {
          partialSums.put(p,
              Lists.<Pair<AggregationBranch,double[]>>newArrayList());
        }
        boolean onlyTag = true;
        for(AggregationBranch b : AggregationBranch.values()) {
          if(tu.hasData(b)) {
            partialSums.get(p).add(Pair.of(b, tu.getCount(b)));
            onlyTag = false;
          }
        }
        if(onlyTag) {
          Pair<Integer, Integer> pair1 = Pair.of(tu.getDocumentId(), tu.getTermId());
          if(!entriesToTag.containsKey(pair1)) {
            entriesToTag.put(pair1, Maps.<AggregationBranch, Integer>newEnumMap(AggregationBranch.class));
          }
          AggregationBranch b1 = AggregationBranch.of(currentGroupingKey.getTermId(),
              currentGroupingKey.getDocId());
          if(!entriesToTag.get(pair1).containsKey(b1)) {
            entriesToTag.get(pair1).put(b1, 0);
          }
          entriesToTag.get(pair1).put(b1, entriesToTag.get(pair1).get(b1) + 1);
        }
      }
      reducer.reduce(currentGroupingKey, tmp, reduceContext);
    }

    // reducer should be given one "aggregating entry" per row, one per column, and one for
    // the global topics sums.
    assertEquals(corpusAsMatrix.numRows() + corpusAsMatrix.numCols() + 1, partialSums.size());

    // every entry to tag should be submitted with each aggregating branch exactly once.
    for(Map.Entry<Pair<Integer, Integer>, EnumMap<AggregationBranch, Integer>> entry
        : entriesToTag.entrySet()) {
      for(AggregationBranch branch : AggregationBranch.values()) {
        if(!entry.getValue().get(branch).equals(1)) {
          fail("Did not find 1 and only 1 entry to be tagged: " + entry.getKey() + " -> " +
            entry.getValue().toString());
        }
      }
    }

    // now have reduced all records
    // make sure you have each of these:
    Map<Pair<Integer,Integer>, EnumMap<AggregationBranch, Integer>> expectedOutputKeys = Maps.newHashMap();
    for(Pair<Integer,Integer> entry : expectedEntries) {
      if(entry.getFirst() == 1444 && entry.getSecond() == 1) {
        entry.getFirst();
      }
      EnumMap<AggregationBranch, Integer> m = Maps.<AggregationBranch, Integer>newEnumMap(AggregationBranch.class);
      for(AggregationBranch b : AggregationBranch.values()) {
        m.put(b, 1);
      }
      expectedOutputKeys.put(entry, m);
    }
    for(Map.Entry<CVBKey, List<CVBTuple>> entry : reduceOutputWriter.getData().entrySet()) {
      CVBKey reduceOutKey = entry.getKey();
      for(CVBTuple reduceOutTuple : entry.getValue()) {
//        assertEquals(reduceOutKey.getTermId(), reduceOutTuple.getTermId());
//        assertEquals(reduceOutKey.getDocId(), reduceOutTuple.getDocumentId());
        for(AggregationBranch branch : AggregationBranch.values()) {
          Pair<Integer, Integer> p = new Pair<Integer, Integer>(
              reduceOutTuple.getDocumentId(),
              reduceOutTuple.getTermId());
          EnumMap<AggregationBranch, Integer> branches = expectedOutputKeys.get(p);
          if(reduceOutTuple.hasData(branch)) {
            branches.put(branch, branches.get(branch) - 1);
          }
        }
      }
    }
    for(Map.Entry<Pair<Integer, Integer>, EnumMap<AggregationBranch, Integer>> entry
        : expectedOutputKeys.entrySet()) {
      for(AggregationBranch branch : AggregationBranch.values()) {
        assertEquals(0, (int)entry.getValue().get(branch));
      }
    }
    for(Pair<Integer,Integer> corpusEntry : expectedEntries) {
      EnumMap<AggregationBranch, Integer> counts = expectedOutputKeys.get(corpusEntry);
      for(AggregationBranch branch : AggregationBranch.values()) {
        assertEquals(0, (int)counts.get(branch));
      }
    }
  }

  private Set<Pair<Integer, Integer>> nonZeroCorpusEntries() throws IOException {
    Set<Pair<Integer, Integer>> set = Sets.newHashSet();
    for(int docId = 0; docId < corpusAsMatrix.numRows(); docId++) {
      Vector doc = corpus(docId);
      for(int termId = 0; termId < doc.size(); termId++) {
        if(doc.get(termId) != 0) {
          set.add(new Pair<Integer, Integer>(docId, termId));
        }
      }
    }
    return set;
  }

  private double corpus(int docId, int termId) throws IOException {
    return corpus(docId).get(termId);
  }

 // @Test
  public void testStage1Mapper() throws Exception {
    Configuration configuration = fs.getConf();
    CVB0Mapper mapper = new CVB0Mapper();
    DummyRecordWriter<CVBKey, CVBTuple> writer = new DummyRecordWriter<CVBKey, CVBTuple>() {
      @Override
      public void write(CVBKey key, CVBTuple tuple) {
        super.write(new CVBKey(key), new CVBTuple(tuple));
      }
    };
    CVB0Mapper.Context context = DummyRecordWriter.build(mapper, configuration, writer);
    mapper.setup(context);
    int termId = 1;
    int docId = 101;
    double count = 2.0;

    mapper.map(key(termId, docId), tuple(termId, docId, count), context);
    Set<CVBKey> expectedKeys = expectedMapOutputKeysFor(termId, docId);
    assertEquals("map output keys incorrect:\n", expectedKeys, writer.getKeys());
    for(CVBKey outKey : writer.getKeys()) {
      List<CVBTuple> expectedValues = expectedMapOutputValuesFor(count, key(termId, docId), outKey);
      List<CVBTuple> values = writer.getValue(outKey);
      verifyMatchingOutputTuples(expectedValues, values);
      assertEquals("wrong # values for " + outKey, expectedValues.size(), values.size());
    }
  }

  private void verifyMatchingOutputTuples(List<CVBTuple> expectedValues, List<CVBTuple> values) {
    for(int i = 0; i < expectedValues.size(); i++) {
      CVBTuple expectedTuple = expectedValues.get(i);
      int foundMatches = 0;
      for(int j = 0; j < values.size(); j++) {
        CVBTuple tuple = values.get(j);
        if(expectedTuple.getDocumentId() == tuple.getDocumentId() &&
           expectedTuple.getTermId() == tuple.getTermId() &&
           expectedTuple.getItemCount() == tuple.getItemCount()) {
          // these could be matches!
          int numBranchesMatch = 0;
          for(AggregationBranch branch : AggregationBranch.values()) {
            if(expectedTuple.hasData(branch) && tuple.hasData(branch) ||
               (!expectedTuple.hasData(branch) && !tuple.hasData(branch))) {
              numBranchesMatch++;
            }
          }
          if(numBranchesMatch == AggregationBranch.values().length) {
            foundMatches++;
          }
        }
      }
      assertEquals("Did not find correct number of matches for: " +
                   Arrays.toString(expectedValues.toArray()) + "\n" +
                   Arrays.toString(values.toArray()), 1, foundMatches);
    }
  }

  private Set<CVBKey> expectedMapOutputKeysFor(int termId, int docId) {
    return Sets.newHashSet(
        key(-1, docId, true, AggregationBranch.of(-1, docId)),
        key(-1, docId, false, AggregationBranch.of(-1, docId)),
        key(termId, -1, true, AggregationBranch.of(termId, -1)),
        key(termId, -1, false, AggregationBranch.of(termId, -1)),
        key(-1, -1, true, AggregationBranch.of(-1, -1)),
        key(-1, -1, false, AggregationBranch.of(-1, -1)));
  }

  private List<CVBTuple> expectedMapOutputValuesFor(double count, CVBKey inputKey, CVBKey outputKey) {
    List<CVBTuple> tuples = Lists.newArrayList();
    // in: (termId, docId, count) leads to out: (-1, docId) -> (termId, docId, count, branch->[ ])
    if(outputKey.isB()) {
      tuples.add(tuple(inputKey.getTermId(), inputKey.getDocId(), count,
               new Pair<AggregationBranch, double[]>(
                 AggregationBranch.of(outputKey.getTermId(), outputKey.getDocId()),
                 new double[]{ 0d })));
    } else {
      tuples.add(tuple(inputKey.getTermId(), inputKey.getDocId(), count));
    }
    return tuples;
  }

  private CVBTuple tuple(int termId, int docId, double itemCount,
      Pair<AggregationBranch, double[]>... counts) {
    CVBTuple tuple = new CVBTuple();
    tuple.setTermId(termId);
    tuple.setDocumentId(docId);
    tuple.setItemCount(itemCount);
    for(Pair<AggregationBranch, double[]> count : counts) {
      tuple.setCount(count.getFirst(), count.getSecond());
    }
    return tuple;
  }

  private CVBKey key(int termId, int docId) {
    return key(termId, docId, true, AggregationBranch.TOPIC_SUM);
  }

  private CVBKey key(int termId, int docId, boolean b, AggregationBranch branch) {
    CVBKey key = new CVBKey();
    key.setTermId(termId);
    key.setDocId(docId);
    key.setB(b);
    key.setBranch(branch);
    return key;
  }


  @Test
  public void testStage1() throws Exception {
    testStage0();
    CVB0Driver driver = new CVB0Driver();
    Path stage1Input = CVB0Driver.stage1InputPath(getTopicModelStatePath(), 0);
    Path stage1Output = CVB0Driver.stage1OutputPath(getTopicModelStatePath(), 0);
    driver.runIterationStage1(fs.getConf(), stage1Input, stage1Output, 1, 1);

    driver.runIterationStage2(fs.getConf(), stage1Input, stage1Output, 1, 1);


    Path partFile = null;
    for(FileStatus status : fs.listStatus(CVB0Driver.stage1OutputPath(getTopicModelStatePath(), 0),
        PathFilters.partFilter())) {
      partFile = status.getPath();
      break;
    }
    SequenceFile.Reader stateReader = new SequenceFile.Reader(fs, partFile, fs.getConf());
    CVBKey key = new CVBKey();
    CVBTuple tuple = new CVBTuple();
    int entryCount = 0;
    CVBKey previousKey = null;
    Map<Pair<Integer,Integer>, EnumMap<AggregationBranch, Integer>> subParts = Maps.newHashMap();
    while(stateReader.next(key, tuple)) {
      entryCount++;
      int termId = key.getTermId();
      int docId = key.getDocId();
      Pair<Integer, Integer> p = new Pair<Integer, Integer>(docId, termId);
      boolean b = key.isB();
      if(docId < 0 || termId < 0) {
        fail("termId and docId should be >= 0");
      } else {
        EnumMap<AggregationBranch, Integer> foundParts = subParts.get(p);
        if(foundParts == null) {
          foundParts = Maps.newEnumMap(AggregationBranch.class);
          subParts.put(p, foundParts);
        }
        for(AggregationBranch branch : AggregationBranch.values()) {
          if(tuple.hasData(branch)) {
            int i = foundParts.containsKey(branch) ? foundParts.get(branch) : 0;
            foundParts.put(branch, i + 1);
          }
        }
        if(foundParts.isEmpty()) {
          fail("found no sub-parts for " + docId + " : " + termId);
        } else if (foundParts.size() > 1) {
          for(AggregationBranch br : AggregationBranch.values()) {
            if(foundParts.containsKey(br) && foundParts.get(br) > 1)
              fail(docId + ":" + termId + " has branches: [" + foundParts + "]" + "\n" + tuple);
          }
        }
      }
      if(previousKey == null) {
        previousKey = new CVBKey();
      } else {
        System.out.println(key.toString() + " => " + tuple.toString());
        System.out.println(previousKey.compareTo(key) >= 0);
      }
      previousKey.setTermId(termId);
      previousKey.setDocId(docId);
      previousKey.setB(b);
    }
    for(int docId = 0; docId < corpusAsMatrix.numRows(); docId++) {
      Vector doc = corpus(docId);
      for(int termId = 0; termId < doc.size(); termId++) {
        Pair<Integer, Integer> p = new Pair<Integer, Integer>(docId, termId);
        if(doc.get(termId) != 0) {
          EnumMap<AggregationBranch, Integer> c = subParts.get(p);
          for(AggregationBranch branch : AggregationBranch.values()) {
            assertEquals(1, (int)c.get(branch));
          }
        } else {
          assertNull(subParts.get(p));
        }
      }
    }
  }

  private Matrix corpusAsMatrix = null;

  private Vector corpus(int docId) throws IOException {
    if(corpusAsMatrix == null) {
      DistributedRowMatrix drm = new DistributedRowMatrix(corpusPath(),
          getTestTempDirPath("matrix"), 0, 0);
      drm.setConf(fs.getConf());
      int numRows = 0;
      int numCols = 0;
      Map<Integer, Vector> vectors = Maps.newHashMap();
      for(MatrixSlice slice : drm) {
        if(numCols == 0) {
          numCols = slice.vector().size();
        }
        numRows = Math.max(slice.index(), numRows);
        vectors.put(slice.index(), slice.vector());
      }
      corpusAsMatrix = new SparseRowMatrix(new int[] {numRows+1, numCols});
      for(Map.Entry<Integer, Vector> e : vectors.entrySet()) {
        corpusAsMatrix.assignRow(e.getKey(), e.getValue());
      }
    }
    return corpusAsMatrix.getRow(docId);
  }

  private Iterable<VectorWritable> points() {
    List<VectorWritable> points = Lists.newArrayList();
    for(double[] vals : corpus) {
      points.add(new VectorWritable(new DenseVector(vals)));
    }
    return points;
  }

  private int numNonZero() throws IOException {
    int n = 0;
    corpus(0);
    for(int docId = 0; docId < corpusAsMatrix.numRows(); docId++) {
      Vector vector = corpus(docId);
      for(int i=0; i<vector.size(); i++) {
        if(vector.get(i) != 0) {
          n++;
        }
      }
    }
    return n;
  }

  private Path topicModelStatePath = null;

  public Path getTopicModelStatePath() throws IOException {
    if(topicModelStatePath == null) {
     topicModelStatePath = getTestTempDirPath("topicState");
    }
    return topicModelStatePath;
  }

  private Path corpusPath = null;

  private Path corpusPath() throws IOException {
    if(true)
      return new Path("/Users/jake/open_src/gitrepo/mahout/examples/bin/work/reuters-out-seqdir-sparse/drm/matrix");
    if(corpusPath == null) {
      corpusPath = getTestTempFilePath("corpus");
    }
    return corpusPath;
  }

  public static Vector sample(Map<Integer, Vector> mixingPercentages, int numSamples) {
    Vector result = null;
    Vector[] topics = new Vector[100];
    int cur = 0;
    for(Map.Entry<Integer, Vector> entry : mixingPercentages.entrySet()) {
      int pct = entry.getKey();
      Vector v = entry.getValue();
      while(pct-- > 0) {
        topics[cur++] = v;
      }
      if(result == null) {
        result = v.like();
      }
    }
    Random random = new Random(1234);
    while(numSamples-- > 0) {
      int topicId = random.nextInt(100);

    }
    return result;
  }

}
