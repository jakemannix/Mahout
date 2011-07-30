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

package org.apache.mahout.vectorizer.document;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.vectorizer.DefaultAnalyzer;
import org.apache.mahout.vectorizer.DocumentProcessor;

import java.io.IOException;
import java.io.StringReader;

public abstract class AbstractTokenizerMapper<K,V> extends Mapper<K, V, Text, StringTuple> {

  public String extractDocument(K key, V value) {
    return value.toString();
  }
  public String extractDocumentId(K key, V value) {
    return key.toString();
  }

  private Analyzer analyzer;
  private Text outputKey = new Text();

  @Override
  protected void map(K key, V value, Context context) throws IOException, InterruptedException {
    String docId = extractDocumentId(key, value);
    String docStr = extractDocument(key, value);
    outputKey.clear();
    outputKey.set(docId.getBytes());
    TokenStream stream = analyzer.reusableTokenStream(docId, new StringReader(docStr));
    CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
    StringTuple document = new StringTuple();
    stream.reset();
    while (stream.incrementToken()) {
      if (termAtt.length() > 0) {
        document.add(new String(termAtt.buffer(), 0, termAtt.length()));
      }
    }
    context.write(outputKey, document);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(context.getConfiguration().get(DocumentProcessor.ANALYZER_CLASS,
          DefaultAnalyzer.class.getName()));
      analyzer = (Analyzer) cl.newInstance();
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }
  }
}
