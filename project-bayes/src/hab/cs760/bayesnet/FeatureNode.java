package hab.cs760.bayesnet;

import com.sun.istack.internal.Nullable;
import hab.cs760.bayesnet.InstanceCounter.Criterion;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hannah on 10/30/17.
 */
public class FeatureNode extends Node {
	public Edge labelNodeEdge;
	public Map<String, Map<String, Map<String, Double>>> probabilities;

	FeatureNode(NominalFeature feature) {
		super(feature);
		this.labelNodeEdge = null;
		probabilities = new HashMap<>();
	}

	@Override
	protected void calculateProbabilitiesForThisNode(List<Instance> instances) {
		Node labelNode = labelNodeEdge.start();
		NominalFeature labelFeature = labelNode.feature;

		Node treeParent = getTreeParent();
		for (String labelFeatureValue : labelFeature.possibleValues) {
			Map<String, Map<String, Double>> block = new HashMap<>();
			probabilities.put(labelFeatureValue, block);

			if (treeParent != null) {
				NominalFeature treeParentFeature = treeParent.feature;

				for (String treeParentFeatureValue : treeParentFeature.possibleValues) {
					Map<String, Double> row = rowWithCriteria(instances, new Criterion
							(labelFeature, labelFeatureValue), new Criterion(treeParentFeature,
							treeParentFeatureValue));
					block.put(treeParentFeatureValue, row);
				}
			} else {
				Map<String, Double> row = rowWithCriteria(instances, new Criterion(labelFeature,
						labelFeatureValue));
				block.put("", row);
			}
		}
	}

	private Map<String, Double> rowWithCriteria(List<Instance> instances, Criterion... criteria) {
		Map<String, Double> row = new HashMap<>();

		for (String featureValue : feature.possibleValues) {
			double probability = InstanceCounter.probabilityOfCriterionGivenCriteria(feature,
					featureValue, instances, criteria);
			row.put(featureValue, probability);
		}
		return row;
	}

	@Nullable
	public Node getTreeParent() {
		for (Edge edge : connectedEdges) {
			if (edge.end() == this) {
				return edge.start();
			}
		}
		return null;
	}

	@Override
	public double probabilityOf(String labelNodeValue, String treeParentValue, String
			thisFeatureValue) {
		if (getTreeParent() != null) {
			assert getTreeParent().feature.possibleValues.contains(treeParentValue);
			return probabilities.get(labelNodeValue).get(treeParentValue).get(thisFeatureValue);
		}
		return probabilities.get(labelNodeValue).get("").get(thisFeatureValue);
	}
}
