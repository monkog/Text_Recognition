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

void mlp(char* fileName)
{
	CvANN_MLP classifier;
	classifier.load(fileName);
	/*
		for (int i = 0; i < testSet.rows; i++)
		classifier.predict(testSet.row(i), results.row(i));*/
}

void bayes(char* fileName)
{
	CvNormalBayesClassifier classifier;
	classifier.load(fileName);
	/*classifier.predict(testSet, &results);*/
}

void r_trees(char* fileName, vector<vector<Mat>> features)
{
	CvRTrees  classifier;
	classifier.load(fileName);

	vector<vector<int>> results;
	for (int i = 0; i < features.size(); i++)
	{
		vector<int> lineResults;
		for (int j = 0; j < features[i].size(); j++)
			lineResults.push_back((int)classifier.predict(features[i][j]));
	}
}

void svm(char* fileName)
{
	CvSVM classifier;
	classifier.load(fileName);

	/*for (int i = 0; i < testSet.rows; i++)
		results.at<int>(i, 0) = ((int)classifier.predict(testSet.row(i)));	*/
}

vector<int> find_white_Pixels(Mat thresholdImage, bool isHorizontal)
{
	vector<int> whitePixels;

	if (isHorizontal)
		for (int j = 0; j < thresholdImage.rows; j++)
		{
			int count = 0;
			for (int i = 0; i < thresholdImage.cols; i++)
				if (thresholdImage.at<uchar>(j, i) > 20) count++;
			whitePixels.push_back(count);
		}
	else
		for (int i = 0; i < thresholdImage.cols; i++)
		{
			int count = 0;
			for (int j = 0; j < thresholdImage.rows; j++)
				if (thresholdImage.at<uchar>(j, i) > 20) count++;
			whitePixels.push_back(count);
		}
	return whitePixels;
}
// Find lines of the text in the image
vector<Mat> find_lines(vector<int> whitePixels, Mat image)
{
	int startIndex = 0, index = 0;
	bool found = true;
	vector<Mat> lines;

	while (found)
	{
		found = false;

		for (int i = index; i < image.rows; i++)
			if (whitePixels[i] != 0)
			{
				startIndex = i;
				found = true;
				break;
			}

		for (int i = startIndex; i < image.rows; i++)
			if (whitePixels[i] == 0)
			{
				index = i;
				break;
			}

		if (index < startIndex)
			index = image.rows - 1;

		lines.push_back(image.rowRange(startIndex, index));
	}
	return lines;
}

vector<Mat> find_words(vector<int> whitePixels, Mat image, int spaceWeight)
{
	int startIndex = 0, index = 0, whiteSpaceCounter = 0;
	bool found = true, foundWord = true;
	vector<Mat> lines;
	int i = 0;

	// Find the begginning of the first word
	for (; i < image.cols; i++)
		if (whitePixels[i] != 0)
		{
			startIndex = i;
			break;
		}

	while (found)
	{
		found = false;

		for (; i < image.cols; i++)
			if (whitePixels[i] == 0)
			{
				if (foundWord)
					foundWord = false;
				found = true;
				break;
			}

		whiteSpaceCounter = 0;
		index = i;
		for (; i < image.cols; i++)
		{
			if (whitePixels[i] != 0)
			{
				if (foundWord)
					startIndex = i;
				break;
			}

			whiteSpaceCounter++;

			if (whiteSpaceCounter > spaceWeight && !foundWord)
			{
				Mat x = image.colRange(startIndex, index);
				lines.push_back(x);
				foundWord = true;
			}
		}
	}
	return lines;
}

vector<Mat> separate_words(vector<int> whiteSpaces, Mat line)
{
	vector<Mat> words;
	int singleSpace = 0;
	int spaceCount = 0;
	int spaceSum = 0;

	for (int i = 0; i < whiteSpaces.size(); i++)
	{
		if (whiteSpaces[i] == 0)
			singleSpace++;
		else if (singleSpace > 0)
		{
			spaceCount++;
			spaceSum += singleSpace;
			singleSpace = 0;
		}
	}

	int spaceWeight = spaceSum / spaceCount;

	return find_words(whiteSpaces, line, spaceWeight);
}
// Calculate the number of white pixels in the matrix
Mat calculateFeatureVector(Mat sample)
{
	Mat features = Mat(FEATURES_VECTOR_SIZE, 1, CV_32F);

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
			features.at<float>(j * 4 + i, 0) = whitePixels;
		}
	return features;
}

vector<vector<Mat>> find_features(vector<vector<Mat>> words)
{
	vector<vector<Mat>> features;
	for (int i = 0; i < words.size(); i++)
	{
		vector<Mat> lineFeatures;
		for (int j = 0; j < words[i].size(); j++)
		{
			Mat sample = words[i][j];
			// Find the contours
			vector<vector<Point>> contours;
			findContours(sample, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

			// Find the bounding box
			vector<vector<Point>> contours_poly(contours.size());
			vector<Rect> boundingBox(contours.size());

			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			// TODO: Sort borders & copy them properly
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			//sort(contours.begin(), contours.end());

			for (int k = 0; k < contours.size(); k++)
			{
				approxPolyDP(Mat(contours[k]), contours_poly[k], 3, true);
				boundingBox[k] = boundingRect(Mat(contours_poly[k]));
			}

			// Resize the contours
			resize(sample.colRange(boundingBox[0].x, boundingBox[0].x + boundingBox[0].width).rowRange(boundingBox[0].y
				, boundingBox[0].y + boundingBox[0].height), sample, boundingBox[0].size());
			int size = max(boundingBox[0].size().width, boundingBox[0].size().height);
			copyMakeBorder(sample, sample, 0, size - boundingBox[0].height, 0, size - boundingBox[0].width, BORDER_ISOLATED);
			resize(sample, sample, Size(32, 32));

			lineFeatures.push_back(calculateFeatureVector(sample));
		}
		features.push_back(lineFeatures);
	}

	return features;
}

int main(int argc, char* argv[])
{
	string imgName = "..\\digits.png";
	cout << "Choose the image to read\n";
	//cin >> imgName;

	Mat image = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
	Mat thresholdImage;
	threshold(image, thresholdImage, 200, 255, THRESH_BINARY_INV);
	vector<Mat> lines = find_lines(find_white_Pixels(thresholdImage, true), thresholdImage);
	vector<vector<Mat>> words;

	for (int i = 0; i < lines.size(); i++)
		words.push_back(separate_words(find_white_Pixels(lines[i], false), lines[i]));

	vector<vector<Mat>> features = find_features(words);

	int model;
	cout << "Choose statistic model:\n" << "1. MLP\n" << "2. Bayes Classifier\n" << "3. R Trees\n" << "4. SVM\n";
	cin >> model;

	string fileName;
	switch (model)
	{
	case MLP:
		fileName = "MLPClassifier.yaml";
		break;
	case Bayes_Classifier:
		fileName = "NormalBayesClassifier.yaml";
		break;
	case R_Trees:
		fileName = "RTreesClassifier.yaml";
		break;
	case Model::SVM:
		fileName = "SVMClassifier.yaml";
		break;
	default:
		break;
	}

	return 0;
}