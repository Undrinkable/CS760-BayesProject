package hab.cs760.test;

import hab.cs760.machinelearning.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Created by hannah on 9/29/17.
 */
public class ArffReaderTests {
	private final static String FILE_PATH = "src/hab/cs760/test/";

	@Test
	void testMakeNumericalFeature() {
		Feature expectedFeature = new NumericFeature("chol", 0);

		Feature actualFeature = ArffReader.makeFeature("@attribute 'chol' real", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute chol real", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute 'chol' numeric", 0);
		assertEquals(expectedFeature, actualFeature);

	}

	@Test
	void testMakeNominalFeature() {
		Feature expectedFeature = new NominalFeature("fbs", 0, "t", "f");

		Feature actualFeature = ArffReader.makeFeature("@attribute 'fbs' { t, f}", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute 'fbs' {t, f }", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute 'fbs' {t,f}", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute 'fbs' {t, f}", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute fbs {t, f}", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute fbs {'t', 'f'}", 0);
		assertEquals(expectedFeature, actualFeature);

		actualFeature = ArffReader.makeFeature("@attribute 'fbs' {'t', 'f'}", 0);
		assertEquals(expectedFeature, actualFeature);


		expectedFeature = new NominalFeature("block_of_affere", 0, "no", "yes");
		actualFeature = ArffReader.makeFeature("@attribute 'block_of_affere' { no, yes}", 0);
		assertEquals(expectedFeature, actualFeature);

		expectedFeature = new NominalFeature("handicapped-infants", 0, "n", "y");
		actualFeature = ArffReader.makeFeature("@attribute 'handicapped-infants' { 'n', 'y'}", 0);
		assertEquals(expectedFeature, actualFeature);
	}

	@Test
	void testInvalidFeature() {
		try {
			ArffReader.makeFeature("@attribute 'chol' { green, red }", 0);
		} catch (IllegalArgumentException e) {
			assertTrue(true);
		} catch (Exception e) {
			Assertions.fail(e);
		}
	}

	@Test
	void testClassLabel1() throws IOException {
		ArffReader arffReader = new ArffReader(FILE_PATH + "lymph_train.arff");
		assertEquals("metastases", arffReader.getPositiveLabel());
		assertEquals("malign_lymph", arffReader.getNegativeLabel());
	}

	@Test
	void testClassLabel2() throws IOException {
		ArffReader arffReader = new ArffReader(FILE_PATH + "vote_train.arff");
		assertEquals("democrat", arffReader.getPositiveLabel());
		assertEquals("republican", arffReader.getNegativeLabel());
	}

	@Test
	void testInstanceCount() throws IOException {
		ArffReader arffReader = new ArffReader(FILE_PATH + "vote_train.arff");
		assertEquals(379, arffReader.instances.size());
	}
	@Test
	void testFeatureCount() throws IOException {
		ArffReader arffReader = new ArffReader(FILE_PATH + "vote_train.arff");
		assertEquals(17, arffReader.featureList.size());
	}

}
