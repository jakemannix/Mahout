package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.Random;

public class CVB0Mapper extends Mapper<CVBKey, CVBTuple, CVBKey, CVBTuple> {
  public static final String NUM_TOPICS = CVB0Mapper.class.getName() + ".numTopics";
  public static final String ETA = CVB0Mapper.class.getName() + ".eta";
  public static final String ALPHA = CVB0Mapper.class.getName() + ".alpha";
  public static final String NUM_TERMS = CVB0Mapper.class.getName() + ".numTerms";
  public static final String RANDOM_SEED = CVB0Mapper.class.getName() + ".seed";

  private int numTopics;
  private double eta;
  private double alpha;
  private double etaTimesNumTerms;
  private long seed;

  private CVBKey outputKey = new CVBKey();
  private CVBTuple outputValue = new CVBTuple();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    numTopics = conf.getInt(NUM_TOPICS, -1);
    eta = conf.getFloat(ETA, 0.1f);
    int numTerms = conf.getInt(NUM_TERMS, -1);
    alpha = conf.getFloat(ALPHA, 0.1f);
    etaTimesNumTerms = eta * numTerms;
    seed = conf.getLong(RANDOM_SEED, 1234L);
    // throw exception on bad config
  }

  @Override
  public void map(CVBKey key, CVBTuple value, Context ctx) throws IOException,
      InterruptedException {
    double[] d = new double[numTopics];
    double total = 0;
    double[] topicTermCounts = value.getCount(AggregationBranch.TOPIC_TERM);
    double[] topicCounts = value.getCount(AggregationBranch.TOPIC_SUM);
    double[] docTopicCounts = value.getCount(AggregationBranch.DOC_TOPIC);
    if(topicTermCounts == null || docTopicCounts == null || topicCounts == null) {
      topicTermCounts = new double[numTopics];
      topicCounts = new double[numTopics];
      docTopicCounts = new double[numTopics];
      initializeCounts(key.getDocId(), key.getTermId(), topicTermCounts, docTopicCounts, topicCounts);
    }
    for(int x = 0; x < numTopics; x++) {
    // p(x | a, i) =~ ((t_ax + eta)/(t_x + eta*W)) * (d_ix + alpha)
      d[x] = (topicTermCounts[x] + eta) / (topicCounts[x] + etaTimesNumTerms);
      d[x] *= (docTopicCounts[x] + alpha);
      total += d[x];
    }
    // L_1 normalize, and then multiply by the item count to get the "pseudo-counts" of
    // C_ai * p(x|a,i) = t_aix = "number of times item a in document i was assigned to topic x"
    for(int x = 0; x < numTopics; x++) {
      d[x] *= (value.getItemCount() / total);
    }
    int termId = key.getTermId();
    int docId = key.getDocId();

    // emit (a, -1, T) : { [t_aix] }
    emitPseudoCountsForAggregating(termId, -1, d, ctx);
    // emit (-1, i, T) : { [t_aix] }
    emitPseudoCountsForAggregating(-1, docId, d, ctx);
    // emit (-1,-1, T) : { [t_aix] }
    emitPseudoCountsForAggregating(-1, -1, d, ctx);

    double itemCount = value.getItemCount();
    // emit (a, -1, F) : { c_ai }
    emitCountForTagging(termId, -1, itemCount, ctx);
    // emit (-1, i, F) : { c_ai }
    emitCountForTagging(-1, docId, itemCount, ctx);
    // emit (-1, -1, F) : { c_ai }
    emitCountForTagging(-1, -1, itemCount, ctx);

  }

  private void initializeCounts(int docId, int termId, double[] topicTermCounts,
      double[] docTopicCounts, double[] topicCounts) {
    Random rand = new Random(seed);
    for(int x = 0; x < numTopics; x++) {
      topicCounts[x] = rand.nextDouble() * etaTimesNumTerms / eta;
    }
    rand = new Random(seed * (termId + 1));
    for(int x = 0; x < numTopics; x++) {
      topicTermCounts[x] = rand.nextDouble() + eta;
    }
    rand = new Random(seed * (docId + 1));
    for(int x = 0; x < numTopics; x++) {
      docTopicCounts[x] = rand.nextDouble();
    }
  }

  private void emitCountForTagging(int termId, int docId, double itemCount, Context ctx)
      throws IOException, InterruptedException {
    outputKey.setTermId(termId);
    outputKey.setDocId(docId);
    outputKey.setB(false);
    outputValue.setItemCount(itemCount);
    for(AggregationBranch branch : AggregationBranch.values()) {
      outputValue.setCount(branch, null);
    }
    ctx.write(outputKey, outputValue);
  }

  private void emitPseudoCountsForAggregating(int termId, int docId, double[] pseudoCounts, Context ctx)
      throws IOException, InterruptedException {
    outputKey.setTermId(termId);
    outputKey.setDocId(docId);
    outputKey.setB(true);
    outputValue.setItemCount(-1);
    outputValue.setCount(AggregationBranch.of(termId, docId), pseudoCounts);
    ctx.write(outputKey, outputValue);
  }

}
