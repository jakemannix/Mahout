package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.Pair;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Random;

public class CVB0Mapper extends Mapper<CVBKey, CVBTuple, CVBKey, CVBTuple> {
  public static final String NUM_TOPICS = CVB0Mapper.class.getName() + ".numTopics";
  public static final String ETA = CVB0Mapper.class.getName() + ".eta";
  public static final String ALPHA = CVB0Mapper.class.getName() + ".alpha";
  public static final String NUM_TERMS = CVB0Mapper.class.getName() + ".numTerms";
  public static final String RANDOM_SEED = CVB0Mapper.class.getName() + ".seed";
  public static final String TEST_SET_PCT = CVB0Mapper.class.getName() + ".testSetFraction";
  public static final String TOPIC_SUM_PARTITIONING_FACTOR = CVB0Mapper.class.getName() + ".partitionFactor";

  private int numTopics;
  private double eta;
  private double alpha;
  private double etaTimesNumTerms;
  private long seed;
  private float testSetFraction;
  private int topicSumPartitioningFactor;

  private CVBKey outputKey = new CVBKey();
  private CVBTuple outputValue = new CVBTuple();
  private double[] topicSum;
  private EnumMap<AggregationBranch, Pair<CVBKey, CVBTuple>> mapsideCombinerCache =
      Maps.newEnumMap(AggregationBranch.class);
  protected CVBInference inference;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    configure(context.getConfiguration());
  }

  public void configure(Configuration conf) {
    numTopics = conf.getInt(NUM_TOPICS, -1);
    topicSum = new double[numTopics];
    eta = conf.getFloat(ETA, 0.1f);
    int numTerms = conf.getInt(NUM_TERMS, -1);
    alpha = conf.getFloat(ALPHA, 0.1f);
    etaTimesNumTerms = eta * numTerms;
    seed = conf.getLong(RANDOM_SEED, 1234L);
    testSetFraction = conf.getFloat(TEST_SET_PCT, 0f);
    topicSumPartitioningFactor = conf.getInt(TOPIC_SUM_PARTITIONING_FACTOR, 10);
    inference = new CVBInference(eta, alpha, numTerms);
    for(AggregationBranch branch : AggregationBranch.values()) {
      mapsideCombinerCache.put(branch, Pair.of(new CVBKey(), new CVBTuple()));
    }
  }

  private int currentTermId;
  private int currentDocId;
  private double currentCount;

  private boolean docInTestSet = false;

  @Override
  public void map(CVBKey key, CVBTuple value, Context ctx) throws IOException,
      InterruptedException {
    if(testSetFraction > 0 && (key.getDocId() % (int)(1/testSetFraction) == 0)) {
      docInTestSet = true;
    } else {
      docInTestSet = false;
    }
    if(!(value.hasAllData())) {
      initializeCounts(key, value);
    }
    double[] d = inference.pTopicGivenTermInDoc(value);
    // Now multiply by the item count to get the "expected-counts" of
    // C_ai * p(x|a,i) = t_aix = "number of times item a in document i was assigned to topic x"
    for(int x = 0; x < numTopics; x++) {
      d[x] *= value.getItemCount();
    }
    int termId = key.getTermId();
    int docId = key.getDocId();
    double itemCount = value.getItemCount();
    currentTermId = termId;
    currentDocId = docId;
    currentCount = itemCount;

    // emit (a, -1, T) : { (-, -, -), [t_aix], -, - }
    emitExpectedCountsForAggregating(termId, -1, d, ctx);
    // emit (-1, i, T) : { (-, -, -), -, [t_aix], - }
    emitExpectedCountsForAggregating(-1, docId, d, ctx);
    // aggregate topicSum
    for(int x = 0; x < numTopics; x++) {
      topicSum[x] += d[x];
    }

    // prepare the output tuple
    outputValue.setTermId(termId);
    outputValue.setDocumentId(docId);
    outputValue.setItemCount(itemCount);

    // for tagging, you don't want any double[].
    outputValue.clearCounts();

    // emit (a, -1, F) : { (a, i, c_ai), -, -, - }
    emitCountForTagging(termId, -1, ctx);
    // emit (-1, i, F) : { (a, i, c_ai), -, -, - }
    emitCountForTagging(-1, docId, ctx);

    // after reduce step:
    // reducer for (a, -1):
    // (a, i, T) : { (-, -, c_ai), [sum_a(t_aix)], -, - }

    // reducer for (-1, i):
    // (a, i, T) : { (-, -, c_ai), - , [sum_i(t_aix)], -}

    // id. mapper
    // reducer:
    // (a, i, T) : { (-, -, c_ai), [sum_a(t_aix)], [sum_i(t_aix)],  []}
    // you get 3 and only 3 tuples for this key.
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // emit the topicSum values to all reducers
    outputKey.setBranch(AggregationBranch.TOPIC_SUM);
    outputKey.setTermId(-1);
    outputKey.setB(true);
    outputValue.clearCounts();
    outputValue.setItemCount(-1);
    outputValue.setCount(AggregationBranch.TOPIC_SUM, topicSum);
    for(int partition = 0; partition < topicSumPartitioningFactor; partition++) {
      outputKey.setDocId(-(1 + partition));
      context.write(outputKey, outputValue);
    }
  }

  private void initializeCounts(CVBKey key, CVBTuple tuple) {
    int termId = key.getTermId();
    int docId = key.getDocId();
    double[] topicCounts = new double[numTopics];
    Random rand = new Random(seed);
    for(int x = 0; x < numTopics; x++) {
      topicCounts[x] = rand.nextDouble() * etaTimesNumTerms / eta;
    }
    tuple.setCount(AggregationBranch.TOPIC_SUM, topicCounts);
    double[] topicTermCounts = new double[numTopics];
    rand = new Random(seed * (termId + 1));
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts[x] = rand.nextDouble() + eta;
    }
    tuple.setCount(AggregationBranch.TOPIC_TERM, topicTermCounts);
    double[] docTopicCounts = new double[numTopics];
    rand = new Random(seed * (docId + 1));
    for(int x = 0; x < numTopics; x++) {
      docTopicCounts[x] = rand.nextDouble();
    }
    tuple.setCount(AggregationBranch.DOC_TOPIC, docTopicCounts);
  }

  private void prepareOutput(int termId, int docId, boolean forAggregating) {
    outputKey.setTermId(termId);
    outputKey.setDocId(docId);
    outputKey.setB(forAggregating);
    outputKey.setBranch(AggregationBranch.of(termId, docId));

    outputValue.setItemCount(currentCount);
    outputValue.setDocumentId(currentDocId);
    outputValue.setTermId(currentTermId);
    outputValue.clearCounts();
  }

  private void emitCountForTagging(int termId, int docId, Context ctx)
      throws IOException, InterruptedException {
    // emit (a, -1, F) | (-1, i, F) | (-1, -1, F) : { (a, i, c_ai), -, -, - }
    prepareOutput(termId, docId, false);
    write(ctx, outputKey, outputValue, false);
  }

  private void emitExpectedCountsForAggregating(int termId, int docId, double[] expectedCounts,
      Context ctx)
      throws IOException, InterruptedException {
    // for documents in the test set, we don't want to emit expected counts to the overall
    // topic model, but for efficiency's sake, we *will* train the doc/topic probabilities for
    // these.  So if the docId passed into this method is < 0, we're emitting for branches
    // TOPIC_SUM or TOPIC_TERM, and for held-out data we *don't* emit.
    if(docInTestSet && docId < 0) {
      return;
    }

    // emit: (a, -1, T) | (-1, i, T) | (-1, -1, T) : { (-, -, -), ... [t_aix] ... }
    // termId and/or docId will be -1
    prepareOutput(termId, docId, true);
    outputValue.setCount(AggregationBranch.of(termId, docId), expectedCounts);
    write(ctx, outputKey, outputValue, true);
  }

  private void write(Context context, CVBKey key, CVBTuple tuple, boolean forAggregation)
      throws IOException, InterruptedException {
    if((forAggregation && tuple.hasData(AggregationBranch.of(key.getTermId(), key.getDocId())))
       || (!forAggregation && !tuple.hasAnyData())) {
      context.write(key, tuple);
    } else {
      throw new IllegalStateException("Wrong Mapper output: (forAggregation: " + forAggregation
                                      + ")" + key + " => " + tuple);
    }
  }

}
