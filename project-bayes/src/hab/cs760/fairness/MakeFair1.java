package hab.cs760.fairness;

import hab.cs760.bayesnet.FeatureNode;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.Map;

/**
 * Implementation of fairness approach #1: "Collapsing the table"
 *
 * Created by hannah on 12/17/17.
 */
public class MakeFair1 extends FairnessStrategy {
	@Override
	protected void makeChildFair(FeatureNode childNode) {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap<>();
		FeatureNode sensitiveNode = (FeatureNode) childNode.getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = childNode.labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb;
		double probSensitiveFeature = -1.0;
		double newProb;
		Map<String, Double> row;

		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);
			for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				row = new HashMap<>();
				block.put(sensitiveFeatureValue, row);
			}

			for (String thisFeatureValue : childNode.feature.possibleValues) {
				newProb = 0.0;
				for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
					probSensitiveFeature = getProbOfSensitiveFeature(childNode, sensitiveNode,
							sensitiveFeatureValue);
					baseProb = childNode.probabilityOf(labelFeatureValue, sensitiveFeatureValue,
							thisFeatureValue);
					newProb += (baseProb * probSensitiveFeature);
				}
				//newProb = newProb / ((double) sensitiveFeature.possibleValues.size());

				for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
					row = block.get(sensitiveFeatureValue);
					//baseProb = childNode.probabilityOf(labelFeatureValue, sensitiveFeatureValue,
					//		thisFeatureValue);
					row.put(thisFeatureValue, newProb);
				}
			}
		}
		childNode.probabilities = newProbabilities;
	}
}
