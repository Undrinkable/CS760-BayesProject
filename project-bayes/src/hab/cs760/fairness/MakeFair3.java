package hab.cs760.fairness;

import hab.cs760.bayesnet.FeatureNode;
import hab.cs760.bayesnet.InstanceCounter;
import hab.cs760.bayesnet.Node;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of fairness approach #3: Weighted Uncertainty, Weighted
 *
 * Created by hannah on 12/17/17.
 */
public class MakeFair3 extends FairnessStrategy {
	private final List<Instance> trainingInstances;

	public MakeFair3(List<Instance> trainingInstances) {
		this.trainingInstances = trainingInstances;
	}

	@Override
	protected void makeChildFair(FeatureNode childNode) {
		Map<String, Map<String, Map<String, Double>>> newProbabilities = new HashMap<>();
		FeatureNode sensitiveNode = (FeatureNode) childNode.getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = childNode.labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double baseProb;
		double probSensitiveFeature;
		double newProb;
		double weightedAverage;

		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			newProbabilities.put(labelFeatureValue, block);

			for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				Map<String, Double> row = new HashMap<>();
				probSensitiveFeature = getProbOfSensitiveFeature(childNode, sensitiveNode,
						sensitiveFeatureValue);
				weightedAverage = getWeightedAverage(childNode, labelFeatureValue,
						sensitiveFeatureValue, sensitiveNode, trainingInstances);

				for (String thisFeatureValue : childNode.feature.possibleValues) {
					baseProb = childNode.probabilityOf(labelFeatureValue, sensitiveFeatureValue,
							thisFeatureValue);
					newProb = (baseProb * probSensitiveFeature) + ((1 - probSensitiveFeature) *
							(weightedAverage));
					row.put(thisFeatureValue, newProb);
				}
				block.put(sensitiveFeatureValue, row);
			}
		}

		childNode.probabilities = newProbabilities;
		normalizeProbTable(childNode);
	}

	private double getWeightedAverage(FeatureNode childNode, String labelFeatureValue, String
			sensitiveFeatureValue, FeatureNode sensitiveFeatureNode, List<Instance> instances) {
		NominalFeature sensitiveFeature = sensitiveFeatureNode.feature;
		double runningProb = 0.0;
		for (String otherSensFeatureVal : sensitiveFeature.possibleValues) {
			if (!otherSensFeatureVal.equals(sensitiveFeatureValue)) {
				for (String thisFeatureValue : childNode.feature.possibleValues) {
					runningProb += (childNode.probabilityOf(labelFeatureValue,
							otherSensFeatureVal, thisFeatureValue) * secondOrderWeights(childNode,
							sensitiveFeatureNode, sensitiveFeatureValue, thisFeatureValue,
							instances));
				}
			}
		}
		return runningProb / (sensitiveFeature.possibleValues.size() - 1);
	}

	/**
	 * Returns the probability of a choice for this feature, knowing that you don't
	 * have a particular sensitive attribute
	 *
	 * @param childNode
	 * @param sensitiveNode
	 * @param sensitiveFeatureValue
	 * @param thisFeatureValue
	 * @return
	 */
	private double secondOrderWeights(FeatureNode childNode, FeatureNode sensitiveNode, String
			sensitiveFeatureValue, String thisFeatureValue, List<Instance> instances) {
		double runningProb = 0.0;
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		InstanceCounter.Criterion crit;

		for (String curSensitiveFeatureValue : sensitiveFeature.possibleValues) {
			if (!sensitiveFeatureValue.equals(curSensitiveFeatureValue)) {
				crit = new InstanceCounter.Criterion(sensitiveFeature, curSensitiveFeatureValue);
				runningProb += InstanceCounter.probabilityOfCriterionGivenCriteria(childNode
						.feature, thisFeatureValue, instances, crit);
			}
		}

		return runningProb;
	}

	private void normalizeProbTable(FeatureNode childNode) {
		FeatureNode sensitiveNode = (FeatureNode) childNode.getTreeParent();
		NominalFeature sensitiveFeature = sensitiveNode.feature;
		Node labelNode = childNode.labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;
		double runningTotal;
		double average;
		double normalizedValue;
		Map<String, Map<String, Double>> block;
		Map<String, Double> row;

		for (String labelFeatureValue : labelFeature.possibleValues) {
			block = childNode.probabilities.get(labelFeatureValue);
			for (String sensitiveFeatureValue : sensitiveFeature.possibleValues) {
				row = block.get(sensitiveFeatureValue);
				runningTotal = 0.0;
				for (String thisFeatureValue : childNode.feature.possibleValues) {
					runningTotal += childNode.probabilityOf(labelFeatureValue,
							sensitiveFeatureValue, thisFeatureValue);
				}

				average = runningTotal / ((double) childNode.feature.possibleValues.size());

				for (String thisFeatureValue : childNode.feature.possibleValues) {
					normalizedValue = childNode.probabilityOf(labelFeatureValue,
							sensitiveFeatureValue, thisFeatureValue) / average;
					row.put(thisFeatureValue, normalizedValue);
				}
			}
		}
	}
}
