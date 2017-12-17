package hab.cs760.test;

import hab.cs760.BayesLearner;
import hab.cs760.bayesnet.BayesMode;
import hab.cs760.bayesnet.InstanceCounter;
import hab.cs760.bayesnet.LabelNode;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.NominalFeature;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import static hab.cs760.bayesnet.Util.conditionalMutualInformation;
import static hab.cs760.test.TestConstants.FILE_PATH;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by hannah on 11/5/17.
 */
public class TanLymphLearnerTest {

	private static final double DELTA = 0.0000000001;
	private BayesLearner learner;

	@BeforeEach
	void setup() {
		learner = new BayesLearner(FILE_PATH + "lymph_train.arff", FILE_PATH + "lymph_test.arff",
				BayesMode
				.TAN);
		learner.buildBayes();
	}

	@Test
	void testStructure() throws FileNotFoundException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "lymph_t_net.txt")).useDelimiter
				("\\Z").next();
		assertEquals(expectedOutput, learner.bayesNetString());
	}

	@Test
	void testConditionalMutualInformation() {
		LabelNode classNode = learner.net.labelNode;
		NominalFeature classLabel = learner.net.labelNode.feature;
		NominalFeature a0 = classNode.connectedEdges.get(0).end().feature;
		NominalFeature a1 = classNode.connectedEdges.get(1).end().feature;
		NominalFeature a2 = classNode.connectedEdges.get(2).end().feature;
		NominalFeature a3 = classNode.connectedEdges.get(3).end().feature;
		NominalFeature a4 = classNode.connectedEdges.get(4).end().feature;
		assertEquals(-1.0, conditionalMutualInformation(a0, a0, classLabel, learner
				.trainInstances));
		assertEquals(0.027291432603321685, conditionalMutualInformation(a0, a1, classLabel, learner
				.trainInstances), DELTA);
		assertEquals(0.03637748761403346, conditionalMutualInformation(a0, a2, classLabel, learner
				.trainInstances), DELTA);
		assertEquals(0.08621954530896751, conditionalMutualInformation(a0, a3, classLabel, learner
				.trainInstances), DELTA);
		assertEquals(0.035706440137138536, conditionalMutualInformation(a0, a4, classLabel, learner
				.trainInstances), DELTA);
	}

	@Test
	void testInstanceCounter() {
		LabelNode classNode = learner.net.labelNode;
		Node a0 = classNode.connectedEdges.get(0).end();
		assertEquals(0.01639344262295082, InstanceCounter.probabilityOfCriteriaGivenLabel
				(classNode.feature, "metastases", learner.trainInstances, new InstanceCounter.Criterion(a0.feature, "normal")), DELTA);

		assertEquals(0.4426229508196721, InstanceCounter.probabilityOfCriteriaGivenLabel(classNode
				.feature, "metastases", learner.trainInstances, new InstanceCounter.Criterion
				(a0.feature, "arched")), DELTA);

		Node a1 = classNode.connectedEdges.get(1).end();
		Node a5 = classNode.connectedEdges.get(5).end();
		assertEquals(0.5172413793103449, InstanceCounter.probabilityOfCriterionGivenCriteria
				(a1.feature, "no", learner.trainInstances, new InstanceCounter.Criterion(classNode
						.feature, "metastases"), new InstanceCounter.Criterion(a5.feature, "no")),
				DELTA);
	}

	@Test
	void testTanLymphPredictions() throws IOException {
		String expectedOutput = new Scanner(new File(FILE_PATH + "lymph_t_predictions.txt"))
				.useDelimiter("\\Z").next();
		assertEquals(expectedOutput, learner.testSetPredictions());
	}


}
