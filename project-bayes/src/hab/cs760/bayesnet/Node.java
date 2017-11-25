package hab.cs760.bayesnet;

import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hannah on 10/30/17.
 */
public abstract class Node {
	public final NominalFeature feature;
	public final List<Edge> connectedEdges;


	Node(NominalFeature feature) {
		this.feature = feature;
		this.connectedEdges = new ArrayList<>();
	}

	void calculateConditionalProbabilities(List<Instance> instances) {
		calculateProbabilitiesForThisNode(instances);
		for (Edge edge : connectedEdges) {
			if (isPointingAway(edge)) {
				edge.end().calculateConditionalProbabilities(instances);
			}
		}
	}

	protected abstract void calculateProbabilitiesForThisNode(List<Instance> instances);

	public abstract double probabilityOf(String labelNodeValue, String treeParentValue, String thisFeatureValue);

	double predict(String labelValue, String treeParentValue, Instance instance) {
		String thisFeatureValue = getFeatureValueForInstance(instance);

		double p = probabilityOf(labelValue, treeParentValue, thisFeatureValue);
		// multiply by all subtrees' probabilities
		for (Edge edge : connectedEdges) {
			if (isPointingAway(edge)) {
				p *= edge.end().predict(labelValue, thisFeatureValue, instance);
			}
		}
		return p;
	}

	private boolean isPointingAway(Edge edge) {
		return edge.end() != this;
	}

	String getFeatureValueForInstance(Instance instance) {
		return instance.getFeatureValue(feature);
	}

	public boolean hasUndirectedEdges() {
		for (Edge edge : connectedEdges) {
			if (edge.isUndirected()) return true;
		}
		return false;
	}
}

