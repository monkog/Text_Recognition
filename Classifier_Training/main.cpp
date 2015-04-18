#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

#define FEATURES_VECTOR_SIZE 16

enum Model
{
	MLP = 1
	, Bayes_Classifier = 2
	, R_Trees = 3
	, SVM = 4
};

void createTrainingSet(vector<vector<int>> allFeatures, vector<vector<int>> allTestFeatures, int numOfSamples, int rows
	, Mat features, Mat labels, Mat testSet)
{
	// Create matrix of features (one row is one feature vector)
	for (int j = 0; j < rows * numOfSamples; j++)
	{
		vector<int> feature = allFeatures[j];
		vector<int> testFeature = allTestFeatures[j];

		for (int i = 0; i < FEATURES_VECTOR_SIZE; i++)
		{
			features.at<float>(j, i) = feature[i];
			testSet.at<float>(j, i) = testFeature[i];
		}
	}

	// Create matrix of results (each row is the answer for each of the rows in features matrix)
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < numOfSamples; j++)
			labels.at<int>(i * numOfSamples + j, 0) = i;
}

void mlp(Mat features, Mat testSet, Mat results, int numOfSamples, int rows)
{
	// 4 rows and 1 col with the type of 32 bits short.
	Mat layer_sizes(4, 1, CV_32SC1);
	layer_sizes.at<int>(0) = 16;   // inputs
	layer_sizes.at<int>(1) = 30;  // hidden
	layer_sizes.at<int>(2) = 30;  // hidden
	layer_sizes.at<int>(3) = rows;   //output

	Mat labels = Mat(results.rows, rows, CV_32FC1);

	for (int j = 0; j < labels.rows / numOfSamples; j++)
		for (int i = 0; i < labels.cols; i++)
			for (int k = 0; k < numOfSamples; k++)
				if (i == j)
					labels.at<float>(j * numOfSamples + k, i) = 1;
				else
					labels.at<float>(j * numOfSamples + k, i) = 0;

	CvANN_MLP classifier(layer_sizes, CvANN_MLP::SIGMOID_SYM, 1, 1);
	classifier.train(features, labels, Mat());

	for (int i = 0; i < testSet.rows; i++)
		classifier.predict(testSet.row(i), results.row(i));

	int wrongAnswers = 0;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < numOfSamples; j++)
			if (results.at<int>(i * numOfSamples + j, 0) != i)
				wrongAnswers++;

	cout << "Wrong answers: " << wrongAnswers << " / " << numOfSamples * rows;

	classifier.save("..\\MLPClassifier.yaml");
}

void bayes(Mat features, Mat labels, Mat testSet, Mat results, int numOfSamples, int rows)
{
	CvNormalBayesClassifier classifier;

	classifier.train(features, labels);
	classifier.predict(testSet, &results);

	int wrongAnswers = 0;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < numOfSamples; j++)
			if (results.at<float>(i * numOfSamples + j, 0) != i)
				wrongAnswers++;

	cout << "Wrong answers: " << wrongAnswers << " / " << numOfSamples * rows;

	classifier.save("..\\NormalBayesClassifier.yaml");
}

void r_trees(Mat features, Mat labels, Mat testSet, Mat results, int numOfSamples, int rows)
{
	CvRTrees  classifier;
	CvRTParams  params(4, // max_depth,
		2, // min_sample_count,
		0.f, // regression_accuracy,
		false, // use_surrogates,
		rows, // max_categories,
		0, // priors,
		false, // calc_var_importance,
		1, // nactive_vars,
		5, // max_num_of_trees_in_the_forest,
		0, // forest_accuracy,
		CV_TERMCRIT_ITER // termcrit_type
		);

	classifier.train(features, CV_ROW_SAMPLE, labels, Mat(), Mat(), Mat(), Mat(), params);

	int wrongAnswers = 0;
	for (int i = 0; i < testSet.rows; i++)
		results.at<int>(i, 0) = ((int)classifier.predict(testSet.row(i)));

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < numOfSamples; j++)
			if (results.at<int>(i * numOfSamples + j, 0) != i)
				wrongAnswers++;

	cout << "Wrong answers: " << wrongAnswers << " / " << testSet.rows;
	classifier.save("..\\RTreesClassifier.yaml");
}

void svm(Mat features, Mat labels, Mat testSet, Mat results, int numOfSamples, int rows)
{
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY; //CvSVM::LINEAR;
	params.degree = 0.5;
	params.gamma = 1;
	params.coef0 = 1;
	params.C = 1;
	params.nu = 0.5;
	params.p = 0;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);

	CvSVM classifier(features, labels, Mat(), Mat(), params);
	int wrongAnswers = 0;
	for (int i = 0; i < testSet.rows; i++)
		results.at<int>(i, 0) = ((int)classifier.predict(testSet.row(i)));

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < numOfSamples; j++)
			if (results.at<int>(i * numOfSamples + j, 0) != i)
				wrongAnswers++;

	cout << "Wrong answers: " << wrongAnswers << " / " << numOfSamples * rows;

	classifier.save("..\\SVMClassifier.yaml");
}

// Calculate the number of white pixels in the matrix
vector<int> calculateFeatureVector(Mat sample)
{
	vector<int> features(FEATURES_VECTOR_SIZE);

	for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
		{
			int whitePixels = 0;
			for (int k = 0; k < 8; k++)
			{
				for (int l = 0; l < 8; l++)
					if (sample.at<uchar>(j * 8 + k, i * 8 + l) > 20)
						whitePixels++;
			}
			features[j * 4 + i] = whitePixels;
		}
	return features;
}

void createFeaturesAndTestSets(Mat thresholdImage, vector<vector<int>>* allFeatures, vector<vector<int>>* allTestFeatures, int rows, int cols, int numOfSamples, int width)
{
	for (int j = 0; j < rows * numOfSamples; j++)
	{
		for (int i = 0; i < cols; i++)
		{
			Mat sample = thresholdImage.colRange(i * width, i * width + width).rowRange(j * width, j * width + width);


			if (j * (cols / 2) + i == 88 || j * (cols / 2) + i - (width / 2) == 88)
				cout << "";
			// Find the contours
			vector<vector<Point>> contours;
			findContours(sample, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			// Find the bounding box
			vector<vector<Point>> contours_poly(contours.size());
			vector<Rect> boundingBox(contours.size());

			for (int k = 0; k < contours.size(); k++)
			{
				approxPolyDP(Mat(contours[k]), contours_poly[k], 3, true);
				boundingBox[k] = boundingRect(Mat(contours_poly[k]));
			}

			if (boundingBox.size() > 0)
			{
				// Resize the contours
				resize(sample.colRange(boundingBox[0].x, boundingBox[0].x + boundingBox[0].width).rowRange(boundingBox[0].y
					, boundingBox[0].y + boundingBox[0].height), sample, boundingBox[0].size());
				int size = max(boundingBox[0].size().width, boundingBox[0].size().height);
				copyMakeBorder(sample, sample, 0, size - boundingBox[0].height, 0, size - boundingBox[0].width, BORDER_ISOLATED);
			}

			resize(sample, sample, Size(32, 32));

			if (i < cols / 2)
				(*allFeatures)[j * (cols / 2) + i] = calculateFeatureVector(sample);
			else
				(*allTestFeatures)[j * (cols / 2) + i - (cols / 2)] = calculateFeatureVector(sample);
		}
	}
}

void parse_training_data(string filename, int model, int width, int cols, int rows, int numOfSamples)
{
	Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	// Separate white and black regions using the threshold function
	// TRESHOLDING OPERATIONS:
	// 0 : Binary
	// 1 : Binary Inverted
	// 2 : Threshold Truncated
	// 3 : Threshold to Zero
	// 4 : Threshold to Zero Inverted

	Mat thresholdImage;
	// src_gray: Our input image
	// dst : output image
	// threshold_value : Values greater than treshold will turn black
	// max_BINARY_value : The value used with the Binary thresholding operations(to set the chosen pixels)
	// threshold_type : One of the 5 thresholding operations.They are listed in the comment section of the function above.
	threshold(image, thresholdImage, 200, 255, THRESH_BINARY_INV);
	vector<vector<int>> allFeatures(cols * rows * numOfSamples / 2);
	vector<vector<int>> allTestFeatures(cols * rows * numOfSamples / 2);
	createFeaturesAndTestSets(thresholdImage, &allFeatures, &allTestFeatures, rows, cols, numOfSamples, width);

	Mat features = Mat(numOfSamples * rows, FEATURES_VECTOR_SIZE, CV_32FC1);
	Mat labels = Mat(numOfSamples * rows, 1, CV_32S);

	Mat testSet = Mat(numOfSamples * rows, FEATURES_VECTOR_SIZE, CV_32FC1);
	Mat results = Mat(numOfSamples * rows, 1, CV_32S);
	createTrainingSet(allFeatures, allTestFeatures, numOfSamples, rows, features, labels, testSet);

	switch (model)
	{
	case MLP:
		mlp(features, testSet, results, numOfSamples, rows);
		break;
	case Bayes_Classifier:
		bayes(features, labels, testSet, results, numOfSamples, rows);
		break;
	case R_Trees:
		r_trees(features, labels, testSet, results, numOfSamples, rows);
		break;
	case Model::SVM:
		svm(features, labels, testSet, results, numOfSamples, rows);
		break;
	default:
		break;
	}
}

void choose_clasifier(bool digits, string filename)
{
	int model;
	cout << "Choose statistic model:\n" << "1. MLP\n" << "2. Bayes Classifier\n" << "3. R Trees\n" << "4. SVM\n";
	cin >> model;

	int width = digits ? 20 : 40, cols = digits ? 100 : 40;

	parse_training_data(filename, model, width, cols, digits ? 10 : 26, 5);
}

void choose_training_file(bool digits)
{
	string filename;
	cout << "Type filename\n";
	//cin >> filename;

	string lettersFile = "..\\letters.png";
	string digitsFile = "..\\digits.png";

	choose_clasifier(digits, digits ? digitsFile : lettersFile);
}

int main(int argc, char* argv[])
{
	int result = 1;
	cout << "Choose training set:\n" << "1. Letters\n" << "2. Digits\n";
	cin >> result;

	bool isDigit = result == 2;
	choose_training_file(isDigit);

	return 0;
}