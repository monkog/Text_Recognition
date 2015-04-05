#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

const int FEATURES_VECTOR_SIZE = 16;

enum Model
{
	MLP = 1
	, Bayes_Classifier = 2
	, R_Trees = 3
	, SVM = 4
};

void mlp(Mat image, vector<vector<int>> allFeatures, int width, int cols, int rows)
{
	CvANN_MLP mlp;
	Mat output;
	mlp.create(image);
	mlp.train(image, output, Mat());
}

vector<int> calculateFeatureVector(Mat sample)
{
	vector<int> features(FEATURES_VECTOR_SIZE);

	for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
		{
			int whitePixels = 0;
			for (int k = 0; k < 8; k++)
				for (int l = 0; l < 8; l++)
					if (sample.at<uchar>(j * 4 + k, i * 4 + l) > 20)
						whitePixels++;
			features[j * 4 + i] = whitePixels;
		}
	return features;
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
	vector<vector<int>> allFeatures(cols * rows * numOfSamples);

	for (int j = 0; j < rows * numOfSamples; j++)
	{
		for (int i = 0; i < cols; i++)
		{
			Mat sample = thresholdImage.colRange(i * width, i * width + width).rowRange(j * width, j * width + width);

			// Find the contours
			vector<vector<Point>> contours;
			findContours(sample, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			// Find the bounding box
			vector<vector<Point>> contours_poly(contours.size());
			vector<Rect> boundingBox(contours.size());

			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundingBox[i] = boundingRect(Mat(contours_poly[i]));
			}

			// Resize the contours
			resize(sample.colRange(boundingBox[0].x, boundingBox[0].x + boundingBox[0].width).rowRange(boundingBox[0].y
				, boundingBox[0].y + boundingBox[0].height), sample, boundingBox[0].size());
			int size = max(boundingBox[0].size().width, boundingBox[0].size().height);
			copyMakeBorder(sample, sample, 0, size - boundingBox[0].height, 0, size - boundingBox[0].width, BORDER_ISOLATED);
			resize(sample, sample, Size(32, 32));
			allFeatures[j * rows * numOfSamples + cols] = calculateFeatureVector(sample);
		}
	}

	switch (model)
	{
	case MLP:
		mlp(thresholdImage, allFeatures, width, cols, rows);
		break;
	case Bayes_Classifier:
		break;
	case R_Trees:
		break;
	case Model::SVM:
		break;
	default:
		break;
	}
}

void choose_clasifier(bool digits, string filename)
{
	int model = 1;
	cout << "Choose statistic model:\n" << "1. MLP\n" << "2. Bayes Classifier\n" << "3. R Trees\n" << "4. SVM\n";
	//cin >> model;

	//int width = 20, cols = 100;
	int width = 40, cols = 40;

	parse_training_data(filename, model, width, cols, digits ? 10 : 26, 5);
}

void choose_training_file(bool digits)
{
	string filename;
	cout << "Type filename\n";
	//cin >> filename;

	string lettersFile = "..\\letters.png";
	string digitsFile = "..\\digits.png";

	choose_clasifier(digits, lettersFile);
}

int main(int argc, char* argv[])
{
	int result = 1;
	cout << "Choose training set:\n" << "1. Letters\n" << "2. Digits\n";
	//cin >> result;

	bool isDigit = result == 2;
	choose_training_file(isDigit);

	return 0;
}