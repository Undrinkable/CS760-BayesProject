package hab.cs760.fairness;

import hab.cs760.bayesnet.FeatureNode;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.Map;

/**
 * Implementation of fairness approach #2: Weighted Uncertainty, Naive
 *
 * Created by hannah on 12/17/17.
 */
public class MakeFair2 extends FairnessStrategy {

	@Override
	protected void makeChildFair(FeatureNode childNode) {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap<>();
		FeatureNode sensitiveNode = (FeatureNode) childNode.getTreeParent();
		if (sensitiveNode == null) {
			throw new IllegalStateException("Tried to make a node without " + "parent fair. Only "
					+ "children of a sensitive feature node should be made fair.");
		}

		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = childNode.labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb;
		double probSensitiveFeature;
		double newProb;

		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);

			for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				Map<String, Double> row = new HashMap<>();
				probSensitiveFeature = getProbOfSensitiveFeature(childNode, sensitiveNode,
						sensitiveFeatureValue);

				for (String thisFeatureValue : childNode.feature.possibleValues) {
					baseProb = childNode.probabilityOf(labelFeatureValue, sensitiveFeatureValue,
							thisFeatureValue);
					newProb = (baseProb * probSensitiveFeature) + ((1 - probSensitiveFeature) *
							(1.0 / childNode.feature.possibleValues.size()));
					row.put(thisFeatureValue, newProb);
				}
				block.put(sensitiveFeatureValue, row);
			}
		}

		childNode.probabilities = newProbabilities;
	}
}
