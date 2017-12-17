package hab.cs760;

import com.sun.istack.internal.Nullable;
import hab.cs760.bayesnet.BayesMode;
import hab.cs760.bayesnet.BayesNet;
import hab.cs760.fairness.FairnessStrategy;
import hab.cs760.fairness.MakeFair1;
import hab.cs760.fairness.MakeFair2;
import hab.cs760.fairness.MakeFair3;
import hab.cs760.machinelearning.ArffReader;
import hab.cs760.machinelearning.Feature;
import hab.cs760.machinelearning.Instance;
import hab.cs760.machinelearning.NominalFeature;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class BayesLearner {
	private final boolean isNaive;
	private final List<Feature> featureList;
	public final List<Instance> trainInstances;
	public BayesNet net;
	private final List<Instance> testInstances;
	private final FairnessStrategy fairnessStrategy;

	public BayesLearner(String trainFile, String testFile, BayesMode mode) {
		ArffReader arffReader = readFile(trainFile);
		featureList = arffReader.getFeatureList();
		trainInstances = arffReader.getInstances();

		arffReader = readFile(testFile);
		testInstances = arffReader.getInstances();

		isNaive = mode == BayesMode.Naive;

		if (mode == BayesMode.FairTAN1) {
			fairnessStrategy = new MakeFair1();
		} else if (mode == BayesMode.FairTAN2) {
			fairnessStrategy = new MakeFair2();
		} else if (mode == BayesMode.FairTAN3) {
			fairnessStrategy = new MakeFair3(trainInstances);
		} else {
			fairnessStrategy = null;
		}
	}

	public static void main(String[] args) {
		if (args.length < 3) {
			errorExitWithMessage("Not enough arguments.");
			return;
		}
		learnBayesAndPrintResults(args);
	}

	private static void learnBayesAndPrintResults(String[] args) {
		String mode = args[2];
		BayesMode bayesMode;
		if (mode.equals("n")) {
			bayesMode = BayesMode.Naive;
		} else if (mode.equals("t")) {
			bayesMode = BayesMode.TAN;
		} else if (mode.equals("f1")) {
			bayesMode = BayesMode.FairTAN1;
		} else if (mode.equals("f2")) {
			bayesMode = BayesMode.FairTAN2;
		} else if (mode.equals("f3")) {
			bayesMode = BayesMode.FairTAN3;
		} else {
			errorExitWithMessage("Mode argument is malformed.");
			return;
		}

		BayesLearner learner = new BayesLearner(args[0], args[1], bayesMode);
		learner.buildBayes();

		System.out.println(learner.bayesNetString());
		System.out.println();
		System.out.println(learner.testSetPredictions());
	}

	private static void errorExitWithMessage(String message) {
		System.out.println(message);
		System.out.println();
		System.out.println(usageString());
		System.exit(1);
	}

	private static String usageString() {
		return "Usage:  bayes <train-set-file> <test-set-file> <mode>\n" +
				" Mode options:\n" +
				"   n  - Naive Bayes\n" +
				"   t  - TAN\n" +
				"   f1 - Fair TAN Approach #1\n" +
				"   f2 - Fair TAN Approach #2\n" +
				"   f3 - Fair TAN Approach #3\n";
	}

	/**
	 * @param fileName name of the file that should be read
	 * @return null if there was an exception when trying to read the file. otherwise, arff
	 * reader object with file contents
	 */
	@Nullable
	private static ArffReader readFile(String fileName) {
		try {
			return new ArffReader(fileName);
		} catch (FileNotFoundException e) {
			errorExitWithMessage("File with name " + fileName + " not found: " + e.getMessage());
		} catch (IOException e) {
			errorExitWithMessage("IOException: " + e.getMessage());
		}
		return null;
	}

	public String testSetPredictions() {
		return testSetPredictions(true);
	}

	private String testSetPredictions(boolean showIndividualPredictions) {
		StringBuilder builder = new StringBuilder();
		NominalFeature classLabel = (NominalFeature) featureList.get(featureList.size() - 1);

		int correctCount = 0;
		for (Instance instance : testInstances) {
			String actualLabel = instance.actualLabel;
			String predictedLabel;
			double prediction = net.predictProbabilityOfFirstLabelValue(instance, isNaive);
			if (prediction < 0.5) {
				predictedLabel = classLabel.possibleValues.get(1);
				prediction = 1 - prediction;
			} else {
				predictedLabel = classLabel.possibleValues.get(0);
			}
			if (actualLabel.equals(predictedLabel)) correctCount++;
			if (showIndividualPredictions) {
				if (builder.length() > 0) builder.append(("\n"));
				builder.append(String.format("%s %s %.12f", predictedLabel, actualLabel,
						prediction));

			}
		}
		if (showIndividualPredictions) {
			builder.append("\n\n");
		}
		builder.append(correctCount);

		return builder.toString();
	}

	private double getAccuracyOnTestset() {
		return (double) Integer.parseInt(testSetPredictions(false)) / testInstances.size();
	}

	public void buildBayes() {
		if (isNaive) {
			net = BayesNet.naiveNet(featureList);
			net.train(trainInstances);
		} else {
			net = BayesNet.treeAugmentedNet(featureList, trainInstances);
			net.train(trainInstances);
			if (fairnessStrategy != null) {
				fairnessStrategy.makeFair(net);
			}
		}
	}

	public String bayesNetString() {
		return net.toString();
	}

}
