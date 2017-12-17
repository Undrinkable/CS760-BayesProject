package hab.cs760.test;

import hab.cs760.BayesLearner;
import hab.cs760.bayesnet.BayesMode;
import hab.cs760.bayesnet.BayesNet;
import hab.cs760.bayesnet.LabelNode;
import hab.cs760.bayesnet.Node;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import static hab.cs760.test.TestConstants.FILE_PATH;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by hannah on 11/4/17.
 */
public class NaiveLymphLearnerTest {

	private static final double DELTA = 0.0000000001;
	private BayesLearner learner;

	@BeforeEach
	void setup() {
		learner = new BayesLearner(FILE_PATH + "lymph_train.arff", FILE_PATH + "lymph_test.arff",
				BayesMode.Naive);
		learner.buildBayes();
	}

	@Test
	void testNaiveLymphNetStructure() throws IOException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "lymph_n_net.txt")).useDelimiter
				("\\Z").next();
		assertEquals(expectedOutput, learner.bayesNetString());
	}

	@Test
	void testNaiveLymphPredictions() throws IOException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "lymph_n_predictions.txt"))
				.useDelimiter("\\Z").next();
		assertEquals(expectedOutput, learner.testSetPredictions());
	}

	@Test
	void testNaiveLymphProbabilities() throws IOException {
		BayesNet net = learner.net;

		LabelNode labelNode = net.labelNode;

		assertEquals(0.5686274509803921, labelNode.probabilityOf("metastases"), DELTA);
		assertEquals(0.43137254901960786, labelNode.probabilityOf("malign_lymph"), DELTA);


		Node lymphatics = net.labelNode.connectedEdges.get(0).end();

		assertEquals(1 / 61.0, naiveBayesProb(lymphatics, "metastases", "normal"), DELTA);
		assertEquals(27 / 61.0, naiveBayesProb(lymphatics, "metastases", "arched"), DELTA);
		assertEquals(20 / 61.0, naiveBayesProb(lymphatics, "metastases", "deformed"), DELTA);
		assertEquals(13 / 61.0, naiveBayesProb(lymphatics, "metastases", "displaced"), DELTA);

		assertEquals(1 / 47.0, naiveBayesProb(lymphatics, "malign_lymph", "normal"), DELTA);
		assertEquals(23 / 47.0, naiveBayesProb(lymphatics, "malign_lymph", "arched"), DELTA);
		assertEquals(11 / 47.0, naiveBayesProb(lymphatics, "malign_lymph", "deformed"), DELTA);
		assertEquals(12 / 47.0, naiveBayesProb(lymphatics, "malign_lymph", "displaced"), DELTA);
	}


	private double naiveBayesProb(Node node, String labelValue, String featureValue) {
		return node.probabilityOf(labelValue, null, featureValue);
	}
}
