package DataFilfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.junit.jupiter.api.Test;

/**
 * FeatureRemover, FeatureTitle and DublicateRemover write rows as the key and Text("") as the value.
 * Hadoop's TextOutputFormat still writes the key/value separator, so every cleaned row ends in a TAB.
 * This is why house_price_analysis.py strips tabs from column names (see review.md).
 */
class OutputFormatTest {
    static class Probe extends TextOutputFormat<Text, Text> {
        static String render(Text k, Text v) throws Exception {
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            new LineRecordWriter<Text, Text>(new DataOutputStream(b)).write(k, v);
            return b.toString("UTF-8");
        }
    }

    @Test
    void emptyTextValueStillLeavesATrailingTab() throws Exception {
        assertEquals("a,b,55.0\t\n", Probe.render(new Text("a,b,55.0"), new Text("")));
    }
}
