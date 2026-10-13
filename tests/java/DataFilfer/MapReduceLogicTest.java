package DataFilfer;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.junit.jupiter.api.Test;

/**
 * Mapper/reducer logic tested with a Mockito mock of the Hadoop Context: no cluster, no files.
 * Each test calls map()/reduce() directly and checks what was passed to context.write().
 */
class MapReduceLogicTest {

    static final String ROW_A = "Owner,0,0,2,BHK,1300.0,1,1,\"Ksfc Layout,Bangalore\",12.96991,77.59796,55.0";

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <C> C ctx(Class<?> contextClass) {
        return (C) mock(contextClass);
    }

    private static final LongWritable K = new LongWritable(0);

    // ---------- DublicateChecker ----------

    @Test
    void checkerMapperEmitsWholeRowWithCountOne() throws Exception {
        Mapper<Object, Text, Text, IntWritable>.Context c = ctx(Mapper.Context.class);
        new DublicateChecker.FilterMapper().map(K, new Text(ROW_A), c);
        verify(c).write(new Text(ROW_A), new IntWritable(1));
    }

    @Test
    void checkerMapperRowWithTooFewFieldsCrashes() {
        // A short/blank line makes words[11] fail: the job dies instead of skipping the line.
        Mapper<Object, Text, Text, IntWritable>.Context c = ctx(Mapper.Context.class);
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> new DublicateChecker.FilterMapper().map(K, new Text("Owner,0,0"), c));
    }

    @Test
    void checkerReducerOnlyEmitsKeysSeenTwiceOrMore() throws Exception {
        Reducer<Text, IntWritable, Text, IntWritable>.Context c = ctx(Reducer.Context.class);
        DublicateChecker.IntSumReducer r = new DublicateChecker.IntSumReducer();
        r.reduce(new Text("single"), List.of(new IntWritable(1)), c);
        r.reduce(new Text("triple"), Arrays.asList(new IntWritable(1), new IntWritable(1), new IntWritable(1)), c);
        verify(c, never()).write(eq(new Text("single")), any());
        verify(c).write(new Text("triple"), new IntWritable(3));
    }

    @Test
    void checkerReducerIsNotSafeAsACombiner_knownBug() throws Exception {
        // The job also registers this class as combiner (DublicateChecker.java:60). A combiner runs
        // per map task on partial counts, so a duplicate whose two copies land in different map
        // tasks is counted 1 + 1 and dropped by the ">= 2" filter before the real reducer sees it.
        // Expected correct behaviour: a combiner must pass partial counts through. Currently fails,
        // so the assertion is wrapped to record it as an expected failure (see review.md).
        Reducer<Text, IntWritable, Text, IntWritable>.Context c = ctx(Reducer.Context.class);
        new DublicateChecker.IntSumReducer().reduce(new Text("dup"), List.of(new IntWritable(1)), c);
        assertThrows(AssertionError.class,
                () -> verify(c).write(new Text("dup"), new IntWritable(1)));
    }

    // ---------- DublicateCounter ----------

    @Test
    void counterMapperEmitsCopiesMinusOne() throws Exception {
        Mapper<LongWritable, Text, Text, IntWritable>.Context c = ctx(Mapper.Context.class);
        new DublicateCounter.FilterMapper().map(K, new Text(ROW_A + "\t3"), c);
        verify(c).write(new Text("Number of replicas is:"), new IntWritable(2));
    }

    @Test
    void counterReducerSumsExtraCopies() throws Exception {
        Reducer<Text, IntWritable, Text, IntWritable>.Context c = ctx(Reducer.Context.class);
        new DublicateCounter.IntSumReducer().reduce(new Text("k"),
                Arrays.asList(new IntWritable(2), new IntWritable(1)), c);
        verify(c).write(new Text("k"), new IntWritable(3));
    }

    // ---------- DublicateRemover ----------

    @Test
    void removerMapperKeysByWholeLine() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        new DublicateRemover.FilterMapper().map(K, new Text(ROW_A), c);
        verify(c).write(new Text(ROW_A), new Text("1"));
    }

    @Test
    void removerReducerWritesEachKeyOnceWhateverTheCount() throws Exception {
        Reducer<Text, Text, Text, Text>.Context c = ctx(Reducer.Context.class);
        new DublicateRemover.IntSumReducer().reduce(new Text(ROW_A),
                Arrays.asList(new Text("1"), new Text("1"), new Text("1")), c);
        verify(c, times(1)).write(any(), any());
        verify(c).write(new Text(ROW_A), new Text(""));
    }

    // ---------- NumberOfReplicatedLines ----------

    @Test
    void replicatedLinesCountsEveryInputLine() throws Exception {
        Mapper<LongWritable, Text, Text, IntWritable>.Context c = ctx(Mapper.Context.class);
        Reducer<Text, IntWritable, Text, IntWritable>.Context rc = ctx(Reducer.Context.class);
        new NumberOfReplicatedLines.FilterMapper().map(K, new Text("anything\t2"), c);
        verify(c).write(new Text("Number of Replicated Lines: "), new IntWritable(1));
        new NumberOfReplicatedLines.IntSumReducer().reduce(new Text("k"),
                Arrays.asList(new IntWritable(1), new IntWritable(1)), rc);
        verify(rc).write(new Text("k"), new IntWritable(2));
    }

    // ---------- EmptyFeatues ----------

    @Test
    void emptyFeaturesReportsLineAndColumnOfEachBlankField() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        EmptyFeatues.FilterMapper m = new EmptyFeatues.FilterMapper();
        m.map(K, new Text("a,b,c"), c);                       // line 1: nothing blank
        m.map(K, new Text("Owner,,0,\"x,y\", ,1"), c);          // line 2: columns 1 and 4 blank
        verify(c).write(new Text("2"), new Text("1"));
        verify(c).write(new Text("2"), new Text("4"));
        verify(c, times(2)).write(any(), any());
    }

    // ---------- FeatureRemover ----------

    @Test
    void featureRemoverConvertsUnitsAndDropsBhkOrRk() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        new FeatureRemover.FilterMapper().map(K, new Text(ROW_A), c);
        String expected = "Owner,0,0,2," + (1300.0 / 10.7639104) + ",1,1,\"Ksfc Layout,Bangalore\","
                + "12.96991,77.59796," + ((55.0 / 94.94) * 100000);
        verify(c).write(new Text(expected), new Text(""));
    }

    @Test
    void featureRemoverOutputHasElevenFieldsAndNoBhkOrRk() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        new FeatureRemover.FilterMapper().map(K, new Text(ROW_A), c);
        org.mockito.ArgumentCaptor<Text> out = org.mockito.ArgumentCaptor.forClass(Text.class);
        verify(c).write(out.capture(), any());
        String[] f = out.getValue().toString().split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
        assertEquals(11, f.length);
        assertFalse(out.getValue().toString().contains("BHK,"));
    }

    @Test
    void featureRemoverCrashesOnAHeaderLineThatIsNotAtIndex29050() {
        // The header is only skipped when it is the 29051st line (lines == 29050), which is where it
        // lands after DublicateRemover sorts the file. Fed the raw file, line 0 is the header.
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        String header = "POSTED_BY,UNDER_CONSTRUCTION,RERA,BHK_NO.,BHK_OR_RK,SQUARE_FT,READY_TO_MOVE,RESALE,ADDRESS,LONGITUDE,LATITUDE,TARGET(PRICE_IN_LACS)";
        assertThrows(NumberFormatException.class,
                () -> new FeatureRemover.FilterMapper().map(K, new Text(header), c));
    }

    @Test
    void featureRemoverSkipsTheLineAtIndex29050() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        FeatureRemover.FilterMapper m = new FeatureRemover.FilterMapper();
        for (int i = 0; i < 29050; i++) {
            m.map(K, new Text(ROW_A), c);
        }
        m.map(K, new Text("POSTED_BY,header,line"), c);       // 29051st line: skipped, no exception
        verify(c, times(29050)).write(any(), any());
    }

    // ---------- FeatureTitle ----------

    @Test
    void featureTitlePrependsHeaderBeforeFirstLineOnly() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        FeatureTitle.FilterMapper m = new FeatureTitle.FilterMapper();
        m.map(K, new Text("row1"), c);
        m.map(K, new Text("row2"), c);
        org.mockito.ArgumentCaptor<Text> out = org.mockito.ArgumentCaptor.forClass(Text.class);
        verify(c, times(3)).write(out.capture(), any());
        assertTrue(out.getAllValues().get(0).toString().startsWith("POSTED_BY,"));
        assertEquals("row1", out.getAllValues().get(1).toString());
        assertEquals("row2", out.getAllValues().get(2).toString());
    }

    // ---------- Linear ----------

    private static String csvRow(double[] x, double y) {
        StringBuilder sb = new StringBuilder();
        for (double v : x) sb.append(v).append(',');
        return sb.append(y).toString();
    }

    /** Feeds header + 26004 noiseless rows (y = 5 + sum(coef*x)) and returns what the mapper wrote. */
    private static java.util.Map<String, String> runLinear() throws Exception {
        Mapper<LongWritable, Text, Text, Text>.Context c = ctx(Mapper.Context.class);
        Linear.FilterMapper m = new Linear.FilterMapper();
        m.map(K, new Text("header"), c);
        Random rnd = new Random(7);
        double[] coef = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int r = 0; r < 26004; r++) {
            double[] x = new double[9];
            double y = 5;
            for (int j = 0; j < 9; j++) { x[j] = rnd.nextDouble() * 10; y += coef[j] * x[j]; }
            m.map(K, new Text(csvRow(x, y)), c);
        }
        org.mockito.ArgumentCaptor<Text> k = org.mockito.ArgumentCaptor.forClass(Text.class);
        org.mockito.ArgumentCaptor<Text> v = org.mockito.ArgumentCaptor.forClass(Text.class);
        verify(c, times(4)).write(k.capture(), v.capture());
        java.util.Map<String, String> res = new java.util.HashMap<>();
        for (int i = 0; i < 4; i++) res.put(k.getAllValues().get(i).toString(), v.getAllValues().get(i).toString());
        return res;
    }

    @Test
    void linearWritesOnlyAfterTheHardcodedRowCountAndReportsTenParameters() throws Exception {
        java.util.Map<String, String> res = runLinear();
        assertEquals("10", res.get("length"));   // intercept + 9 features
        assertTrue(res.containsKey("RSquared"));
        assertTrue(res.containsKey("AdjustedRSquared"));
    }

    @Test
    void linearRecoversTrueCoefficientsOnNoiselessData_knownBug() throws Exception {
        // Expected: intercept 5 exactly. Actual: the arrays are sized 26005 but only 26004 rows are
        // filled (Linear.java:23-24 vs :41), so one all-zero row (x=0, y=0) is regressed too and
        // pulls the fit off. Wrapped as an expected failure; see review.md.
        java.util.Map<String, String> res = runLinear();
        double intercept = Double.parseDouble(res.get("estimateRegressionParameters").split(",")[0].trim());
        assertThrows(AssertionError.class, () -> assertEquals(5.0, intercept, 1e-6));
        assertEquals(5.0, intercept, 0.01);      // close, but not exact
    }
}
