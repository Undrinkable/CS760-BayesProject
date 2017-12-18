package hab.cs760.test;

import hab.cs760.bayesnet.BayesMode;
import hab.cs760.bayesnet.BayesNet;
import hab.cs760.bayesnet.LabelNode;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.Feature;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Created by hannah on 10/30/17.
 */
public class NaiveBayesTest1 {
	private static final double DELTA = 0.00000001;
	private List<Feature> features;
	private List<Instance> trainInstances;

	@BeforeEach
	void setup() {
		features = Arrays.asList(
				new NominalFeature("f0", 0, "n", "y"),
				new NominalFeature("f1", 1, "n", "y"),
				new NominalFeature("class", 2, "r", "d"));
		trainInstances = Arrays.asList(
				instanceWithFeatures("n", "n", "r"),
				instanceWithFeatures("y", "n", "d"),
				instanceWithFeatures("y", "y", "d"));
	}

	private Instance instanceWithFeatures(String f0, String f1, String label) {
		Instance instance = new Instance();
		instance.addFeature((NominalFeature) features.get(0), f0);
		instance.addFeature((NominalFeature) features.get(1), f1);
		instance.addFeature((NominalFeature) features.get(2), label);
		return instance;
	}


	@Test
	void testProbabilities() {
		BayesNet net = BayesNet.naiveNet(features);
		net.train(trainInstances);
		LabelNode node = net.labelNode;

		assertEquals(0.4, node.probabilityOf("r"), DELTA);
		assertEquals(0.6, node.probabilityOf("d"), DELTA);

		Node f0 = net.labelNode.connectedEdges.get(0).end();
		assertEquals(0.33333333, f0.probabilityOf("r",null,"y"), DELTA);
		assertEquals(0.66666666, f0.probabilityOf("r",null,"n"), DELTA);
		assertEquals(0.75, f0.probabilityOf("d",null,"y"), DELTA);
		assertEquals(0.25, f0.probabilityOf("d",null,"n"), DELTA);

		Node f1 = net.labelNode.connectedEdges.get(1).end();
		assertEquals(0.33333333, f1.probabilityOf("r",null,"y"), DELTA);
		assertEquals(0.66666666, f1.probabilityOf("r",null,"n"), DELTA);
		assertEquals(0.5, f1.probabilityOf("d",null,"y"), DELTA);
		assertEquals(0.5, f1.probabilityOf("d",null,"n"), DELTA);
	}

	@Test
	void testInferenceByEnumeration() {
		BayesNet net = BayesNet.naiveNet(features);
		net.train(trainInstances);

		Instance testInstance = instanceWithFeatures("y","y","r");
		assertEquals(0.1649484522, net.predictProbabilityOfFirstLabelValue(testInstance, BayesMode.Naive),
				0.001);
	}

}
