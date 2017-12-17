package hab.cs760.bayesnet;

import hab.cs760.machinelearning.Feature;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

/**
 * Created by hannah on 10/30/17.
 */
public class BayesNet {
	public LabelNode labelNode;
	public FeatureNode sensitiveNode;

	public static BayesNet naiveNet(List<Feature> featureList) {
		NominalFeature classLabel = (NominalFeature) featureList.get(featureList.size() - 1);
		featureList = featureList.subList(0, featureList.size() - 1);

		BayesNet net = new BayesNet();
		net.labelNode = new LabelNode(classLabel);
		for (Feature feature : featureList) {
			FeatureNode childNode = new FeatureNode((NominalFeature) feature);
			Edge edge = Edge.directedEdge(net.labelNode, childNode);
			net.labelNode.connectedEdges.add(edge);
			childNode.labelNodeEdge = edge;

			if (feature == featureList.get(0)) {
				net.sensitiveNode = childNode;
			}
		}
		return net;
	}

	public static BayesNet treeAugmentedNet(List<Feature> featureList, List<Instance>
			trainInstances) {
		// create naive net and calculate probability tables
		BayesNet net = naiveNet(featureList);
		net.train(trainInstances);

		// 1. add edges between features
		for (Edge firstEdge : net.labelNode.connectedEdges) {
			Node child1 = firstEdge.end();
			for (Edge secondEdge : net.labelNode.connectedEdges) {
				Node child2 = secondEdge.end();
				if (child1 != child2) {
					// add edge if there isn't already one
					boolean alreadyExists = false;
					for (Edge edge : child1.connectedEdges) {
						List<Node> nodes = edge.connectedNodes();
						if (nodes.contains(child1) && nodes.contains(child2)) {
							alreadyExists = true;
						}
					}
					if (!alreadyExists) {
						Edge edge = Edge.undirectedEdge(child1, child2);
						child1.connectedEdges.add(edge);
						child2.connectedEdges.add(edge);
					}
				}
			}
		}

		// 2. compute weights for edges
		for (Edge edge : net.labelNode.connectedEdges) {
			Node featureNode1 = edge.end();
			for (Edge featureToFeatureEdge : featureNode1.connectedEdges) {
				Node featureNode2 = featureToFeatureEdge.node1 == featureNode1 ?
						featureToFeatureEdge.node2 : featureToFeatureEdge.node1;
				featureToFeatureEdge.weight = Util.conditionalMutualInformation
						(featureNode1.feature, featureNode2.feature, net.labelNode.feature,
								trainInstances);
			}
		}

		// 3. find maximum weight spanning tree
		List<Node> treeNodes = new ArrayList<>();
		// add node for first feature (first feature node should be root by homework specification)
		treeNodes.add(net.labelNode.connectedEdges.get(0).end());

		List<Edge> treeEdges = new ArrayList<>();
		while (treeNodes.size() < net.labelNode.connectedEdges.size()) {
			Edge maxWeightEdge = null;
			for (Node node : treeNodes) {
				for (Edge edge : node.connectedEdges) {
					if (maxWeightEdge == null || maxWeightEdge.weight < edge.weight) {
						if (!treeEdges.contains(edge) && (!treeNodes.contains(edge.node1) ||
								!treeNodes.contains(edge.node2))) {
							maxWeightEdge = edge;
						}
					}
				}
			}
			if (maxWeightEdge == null) {
				throw new IllegalStateException("Ran out of edges before all nodes were in tree");
			}
			treeEdges.add(maxWeightEdge);
			for (Node node : maxWeightEdge.connectedNodes()) {
				if (!treeNodes.contains(node)) treeNodes.add(node);
			}
		}

		// 4. assign edge directions (and delete edges we don't want in the tree)
		Queue<Node> remainingNodes = new ArrayDeque<>();
		remainingNodes.add(net.labelNode.connectedEdges.get(0).end());
		while (!remainingNodes.isEmpty()) {
			Node currentNode = remainingNodes.remove();
			List<Edge> edgesToRemove = new ArrayList<>();
			for (Edge edge : currentNode.connectedEdges) {
				if (!treeEdges.contains(edge)) {
					edgesToRemove.add(edge);
				} else {
					if (edge.isUndirected()) {
						edge.setStart(currentNode);
					}
					if (!remainingNodes.contains(edge.end()) && edge.end().hasUndirectedEdges()) {
						remainingNodes.add(edge.end());
					}
				}
			}
			for (Edge edge : edgesToRemove) {
				edge.node1.connectedEdges.remove(edge);
				edge.node2.connectedEdges.remove(edge);
			}
		}
		return net;

	}


	public void train(List<Instance> instances) {
		labelNode.calculateConditionalProbabilities(instances);
	}

	@Override
	public String toString() {
		return labelNode.toString();
	}

	public double predictProbabilityOfFirstLabelValue(Instance instance, boolean isNaive) {
		instance.setClassLabel(0);
		double firstValueProbability = labelNode.predict(instance, isNaive);

		instance.setClassLabel(1);
		double secondValueProbability = labelNode.predict(instance, isNaive);

		instance.revertClassLabel();

		return firstValueProbability / (firstValueProbability + secondValueProbability);
	}

}
