package org.apache.mahout.analysis;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharTokenizer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.util.Version;

import java.io.Reader;

public class PigAnalyzer extends Analyzer {
  @Override public TokenStream tokenStream(String s, Reader reader) {
    return new CharTokenizer(Version.LUCENE_31, reader) {
      @Override protected boolean isTokenChar(int c) {
        return c == '_' || Character.isLetterOrDigit(c);
      }
    };
  }
}
