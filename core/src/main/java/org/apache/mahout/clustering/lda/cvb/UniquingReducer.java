package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.Iterator;

public class UniquingReducer<K,V> extends Reducer<K,V,K,V> {

  @Override
  public void reduce(K key, Iterable<V> values, Context context)
      throws IOException, InterruptedException {
    Iterator<V> it = values.iterator();
    if(it.hasNext()) {
      context.write(key, it.next());
    }
  }

}
