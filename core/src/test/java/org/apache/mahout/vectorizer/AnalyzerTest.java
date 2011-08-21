package org.apache.mahout.vectorizer;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharTokenizer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.io.Reader;
import java.io.StringReader;

public class AnalyzerTest extends MahoutTestCase {

  @Test
  public void testTokenize() throws Exception {
    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_31);
    TokenStream stream = analyzer.tokenStream("",
        new StringReader( "{(hellobek),(shitmydadsays),(Nano_Story),(josh1jacobs),(michaelepstein),"
                          + "(jeremy),(fredwilson),(bjfogg),(elatable),(peayvineyards),(brynn),"
                          + "(jasonmonberg),(MPosada),(mattccompton),(anniehips),(stewart),"
                          + "(ryanchris),(danarkind),(jeff),(Joshmedia),(chadphillipsnyc),"
                          + "(chanover),(tonysphere),(benrigby),(fredtrotter),(biz),(guido),"
                          + "(kig),(florian),(nivi),(jlanzone),(cricketwardein),(jeffrey),"
                          + "(Adam),(ddukes),(beebe),(dlanham),(trueventures),(azaaza),"
                          + "(howardlindzon),(phclouin),(SarahM),(mlaaker),(Emergency_In_SF),"
                          + "(dsamuel),(beninato),(bfeld),(robhayes),(e_ramirez),(shissla),"
                          + "(healthmonth),(cvorkink),(FortKnoxFive),(mrs_joey),(edcampaniello),"
                          + "(crystal),(RonKurti),(RealTracyMorgan),(adaugelli),(timusica),"
                          + "(InfectiousArt),(veen),(noah),(tempo),(rayreadyray),(mklaurence),"
                          + "(sippey),(simonsmith001),(Caterina),(thepartycow),(twang),(r0bl0rd),"
                          + "(natekoechley),(joshk),(bonforte),(ev),(suzhoward),(rahbean),(sixwing),"
                          + "(naval),(tonystubblebine),(mikeyion)}"));
    CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
    System.out.println("Now about to print!");
    while (stream.incrementToken()) {
      if (termAtt.length() > 0) {
        System.out.println(new String(termAtt.buffer(), 0, termAtt.length()));
      }
    }
  }

  private static final class TestAnalyzer extends CharTokenizer {
    private final Analyzer whitespaceAnalyzer = new WhitespaceAnalyzer(Version.LUCENE_31);
    private static final Character COMMA = ',';
    public TestAnalyzer(Reader input) {
      super(Version.LUCENE_31, input);
    }

    @Override protected boolean isTokenChar(int c) {
      return !Character.isWhitespace(c) && !COMMA.equals(c);
    }
  }

}
