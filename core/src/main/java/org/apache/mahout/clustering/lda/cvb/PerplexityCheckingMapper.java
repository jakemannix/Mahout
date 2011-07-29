package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class PerplexityCheckingMapper extends Mapper<CVBKey, CVBTuple, CVBKey, CVBTuple> {

  public static final String NUM_TOPICS = CVB0Mapper.class.getName() + ".numTopics";
  public static final String ETA = CVB0Mapper.class.getName() + ".eta";
  public static final String ALPHA = CVB0Mapper.class.getName() + ".alpha";
  public static final String NUM_TERMS = CVB0Mapper.class.getName() + ".numTerms";
  public static final String RANDOM_SEED = CVB0Mapper.class.getName() + ".seed";
  public static final String TEST_SET_PCT = CVB0Mapper.class.getName() + ".testSetFraction";

  private int numTopics;
  private double eta;
  private double alpha;
  private double etaTimesNumTerms;
  private long seed;
  private float testSetFraction;

  private CVBKey outputKey = new CVBKey();
  private CVBTuple outputValue = new CVBTuple();
  protected CVBInference inference;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    configure(context.getConfiguration());
  }

  public void configure(Configuration conf) {
    numTopics = conf.getInt(NUM_TOPICS, -1);
    eta = conf.getFloat(ETA, 0.1f);
    int numTerms = conf.getInt(NUM_TERMS, -1);
    alpha = conf.getFloat(ALPHA, 0.1f);
    etaTimesNumTerms = eta * numTerms;
    seed = conf.getLong(RANDOM_SEED, 1234L);
    testSetFraction = conf.getFloat(TEST_SET_PCT, 0f);
    inference = new CVBInference(eta, alpha, numTerms);
  }

  @Override
  public void map(CVBKey key, CVBTuple value, Context context)
      throws IOException, InterruptedException {
    if(testSetFraction > 0 && (key.getDocId() % (int)(1/testSetFraction) == 0)) {
      double[] gamma = inference.gammaTopicGivenTermInDoc(value);
      key.setB(true);
      key.setTermId(-1);
      key.setDocId(key.getDocId());
      value.setDocumentId(key.getDocId());
      value.setCount(AggregationBranch.DOC_TOPIC, gamma);
      context.write(key, value);
    }
  }

}
