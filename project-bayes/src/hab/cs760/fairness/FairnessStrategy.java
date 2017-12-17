package hab.cs760.fairness;

import hab.cs760.bayesnet.BayesNet;
import hab.cs760.bayesnet.Edge;
import hab.cs760.bayesnet.FeatureNode;
import hab.cs760.machinelearning.NominalFeature;

import java.util.List;

/**
 * Created by hannah on 12/17/17.
 */
public abstract class FairnessStrategy {

	public void makeFair(BayesNet net) {
		List<Edge> sensitiveNodeEdges = net.sensitiveNode.connectedEdges;
		FeatureNode childNode;
		for (Edge edge : sensitiveNodeEdges) {
			if (net.sensitiveNode.isPointingAway(edge)) {
				childNode = (FeatureNode) edge.end();
				makeChildFair(childNode);
			}
		}
	}

	protected double getProbOfSensitiveFeature(FeatureNode childNode, FeatureNode sensitiveNode,
											String choice) {
		double runningProb = 0.0;
		NominalFeature labelFeature = childNode.labelNodeEdge.start().feature;
		for (String labelFeatureValue : labelFeature.possibleValues) {
			runningProb += sensitiveNode.probabilityOf(labelFeatureValue, "", choice);
		}
		return runningProb / labelFeature.possibleValues.size();
	}

	protected abstract void makeChildFair(FeatureNode childNode);
}
