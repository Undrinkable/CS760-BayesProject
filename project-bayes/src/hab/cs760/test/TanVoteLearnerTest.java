package hab.cs760.test;

import hab.cs760.BayesLearner;
import hab.cs760.bayesnet.BayesMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import static hab.cs760.test.TestConstants.FILE_PATH;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by hannah on 11/5/17.
 */
public class TanVoteLearnerTest {
	private BayesLearner learner;

	@BeforeEach
	void setup() {
		learner = new BayesLearner(FILE_PATH + "vote_train.arff", FILE_PATH + "vote_test.arff",
				BayesMode
				.TAN);
		learner.buildBayes();
	}

	@Test
	void testStructure() throws FileNotFoundException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "vote_t_net.txt")).useDelimiter
				("\\Z").next();
		assertEquals(expectedOutput, learner.bayesNetString());
	}

	@Test
	void testTanVotePredictions() throws IOException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "vote_t_predictions.txt"))
				.useDelimiter("\\Z").next();
		assertEquals(expectedOutput, learner.testSetPredictions());
	}
}
