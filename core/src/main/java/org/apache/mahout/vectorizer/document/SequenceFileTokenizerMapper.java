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

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;

import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.vectorizer.DefaultAnalyzer;
import org.apache.mahout.vectorizer.DocumentProcessor;

/**
 * Tokenizes a text document and outputs tokens in a StringTuple
 */
public class SequenceFileTokenizerMapper extends AbstractTokenizerMapper<Text, Text> {

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    analyzer = ClassUtils.instantiateAs(context.getConfiguration().get(DocumentProcessor.ANALYZER_CLASS,
                                                                       DefaultAnalyzer.class.getName()),
                                        Analyzer.class);
  }
}
