package hab.cs760.bayesnet;

import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.HashMap;
import java.util.List;

/**
 * Created by hannah on 10/30/17.
 */
public class LabelNode extends Node {
	private final HashMap<String, Double> probabilities;

	public LabelNode(NominalFeature feature) {
		super(feature);
		probabilities = new HashMap<>();
	}

	@Override
	protected void calculateProbabilitiesForThisNode(List<Instance> instances) {
		for (String featureValue : feature.possibleValues) {
			double probability = InstanceCounter.probabilityOfCriteriaGivenLabel(null, null,
					instances, new InstanceCounter.Criterion(feature, featureValue));
			probabilities.put(featureValue, probability);
		}
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for (Edge child : connectedEdges) {
			Node featureNode = child.end();
			if (builder.length() > 0) builder.append("\n");
			builder.append(featureNode.feature.name);
			builder.append(" ");
			for (Edge edge : featureNode.connectedEdges) {
				if (edge.end() == featureNode) {
					builder.append(edge.start().feature.name);
					builder.append(" ");
				}
			}
			builder.append(((FeatureNode) featureNode).labelNodeEdge.start().feature.name);
		}
		return builder.toString();
	}

	@Override
	public double probabilityOf(String labelNodeValue, String treeParentValue, String
			thisFeatureValue) {
		return probabilityOf(thisFeatureValue);
	}

	public double probabilityOf(String thisFeatureValue) {
		return probabilities.get(thisFeatureValue);
	}

	double predict(Instance instance, BayesMode mode) {
		String thisFeatureValue = getFeatureValueForInstance(instance);
		if (mode == BayesMode.Naive) {
			return super.predict(thisFeatureValue, null, instance, mode);
		} else {
			double p = probabilityOf(thisFeatureValue);
			// in a TAN, we need to multiply the label node probability by the probability term
			// for the tree. the tree is rooted at the first feature as per hw4 specification.
			return p * connectedEdges.get(0).end().predict(thisFeatureValue, null, instance, mode);
		}
	}
}
