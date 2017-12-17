package hab.cs760.machinelearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by hannah on 10/30/17.
 */
public class ArffReader {
	public final List<Feature> featureList;
	public final List<Instance> instances;

	public ArffReader(String datafileName) throws IOException {
		featureList = new ArrayList<>();
		instances = new ArrayList<>();
		readFeaturesAndInstances(datafileName);
	}

	public static Feature makeFeature(String line, int index) {
		String[] splitLine = line.split(" ");
		String featureName = splitLine[1].replace("\'", "");
		if (featureName.endsWith(".")) {
			featureName = featureName.substring(0, featureName.length()-1);
		}
		featureName = featureName.trim();

		Feature feature;
		if (!splitLine[2].startsWith("{") && (splitLine[2].contains("numeric") || line.contains
				("real"))) {
			feature = new NumericFeature(featureName, index);
		} else {
			int startIndex = line.indexOf('{');
			int endIndex = line.indexOf('}');
			feature = new NominalFeature(featureName, index, line.substring(startIndex +1 ,
					endIndex));
		}

		return feature;
	}

	private void readFeaturesAndInstances(String fileName) throws IOException {
		FileReader fileReader = new FileReader(fileName);
		BufferedReader bufferedReader = new BufferedReader(fileReader);

		String line = bufferedReader.readLine();

		while (line != null) {
			processLine(line);
			line = bufferedReader.readLine();
		}

		bufferedReader.close();

	}

	private void processLine(String line) {
		String annotation = line.split(" ")[0];
		annotation = annotation.toLowerCase();
		if (annotation.startsWith("%")) return;
		if ("@relation".equals(annotation)) return;
		if ("@data".equals(annotation)) return;

		if ("@attribute".equals(annotation)) {
			featureList.add(makeFeature(line, featureList.size()));
		} else {
			instances.add(Instance.makeInstance(line, featureList));
		}
	}

	public List<Feature> getFeatureList() {
		return featureList;
	}

	public List<Instance> getInstances() {
		return instances;
	}

	public String getPositiveLabel() {
		return ((NominalFeature)featureList.get(featureList.size()-1)).possibleValues.get(0);
	}

	public String getNegativeLabel() {
		return ((NominalFeature)featureList.get(featureList.size()-1)).possibleValues.get(1);
	}
}
