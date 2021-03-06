#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

#define FEATURES_VECTOR_SIZE 16

vector<vector<Rect>> m_words;
vector<tuple<int, int>> m_lines;
vector<vector<vector<Rect>>> m_letters;

enum Model
{
	MLP = 1
	, Bayes_Classifier = 2
	, R_Trees = 3
	, SVM = 4
};

vector<vector<vector<int>>> mlp(string fileName, vector<vector<vector<Mat>>> features)
{
	CvANN_MLP classifier;
	classifier.load(fileName.c_str());

	vector<vector<vector<int>>> results;
	for (int i = 0; i < features.size(); i++)
	{
		vector<vector<int>> lineResults;
		for (int j = 0; j < features[i].size(); j++)
		{
			vector<int> wordResults;
			for (int k = 0; k < features[i][j].size(); k++)
			{
				Mat result;
				classifier.predict(features[i][j][k], result);
				float max = 0;
				int maxIndex = 0;
				for (int l = 0; l < result.cols; l++)
					if (result.at<float>(0, l) > max)
					{
						max = result.at<float>(0, l);
						maxIndex = l;
					}
				wordResults.push_back(maxIndex);
			}
			lineResults.push_back(wordResults);
		}
		results.push_back(lineResults);
	}
	return results;
}

vector<vector<vector<int>>> bayes(string fileName, vector<vector<vector<Mat>>> features)
{
	CvNormalBayesClassifier classifier;
	classifier.load(fileName.c_str());

	Mat result = Mat(1, 1, CV_32FC1);

	vector<vector<vector<int>>> results;
	for (int i = 0; i < features.size(); i++)
	{
		vector<vector<int>> lineResults;
		for (int j = 0; j < features[i].size(); j++)
		{
			vector<int> wordResults;
			for (int k = 0; k < features[i][j].size(); k++)
			{
				classifier.predict(features[i][j][k], &result);
				wordResults.push_back(result.at<float>(0, 0));
			}
			lineResults.push_back(wordResults);
		}
		results.push_back(lineResults);
	}
	return results;
}

vector<vector<vector<int>>> r_trees(string fileName, vector<vector<vector<Mat>>> features)
{
	CvRTrees  classifier;
	classifier.load(fileName.c_str());

	vector<vector<vector<int>>> results;
	for (int i = 0; i < features.size(); i++)
	{
		vector<vector<int>> lineResults;
		for (int j = 0; j < features[i].size(); j++)
		{
			vector<int> wordResults;
			for (int k = 0; k < features[i][j].size(); k++)
				wordResults.push_back((int)classifier.predict(features[i][j][k]));
			lineResults.push_back(wordResults);
		}
		results.push_back(lineResults);
	}
	return results;
}

vector<vector<vector<int>>> svm(string fileName, vector<vector<vector<Mat>>> features)
{
	CvSVM classifier;
	classifier.load(fileName.c_str());

	vector<vector<vector<int>>> results;
	for (int i = 0; i < features.size(); i++)
	{
		vector<vector<int>> lineResults;
		for (int j = 0; j < features[i].size(); j++)
		{
			vector<int> wordResults;
			for (int k = 0; k < features[i][j].size(); k++)
				wordResults.push_back((int)classifier.predict(features[i][j][k]));
			lineResults.push_back(wordResults);
		}
		results.push_back(lineResults);
	}
	return results;
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

	int lastWhiteRow = image.rows;

	for (int i = image.rows - 1; i >= 0; i--)
		if (whitePixels[i] != 0)
		{
			lastWhiteRow = i;
			break;
		}

	while (found && index < lastWhiteRow)
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
		m_lines.push_back(tuple<int, int>(startIndex, index));
	}
	return lines;
}

vector<Mat> find_words(vector<int> whitePixels, Mat image, int spaceWeight, tuple<int, int> lineRange)
{
	int startIndex = 0, index = 0, whiteSpaceCounter = 0;
	bool found = true, foundWord = true;
	vector<Mat> lines;
	vector<Rect> wordRects;
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
				wordRects.push_back(Rect(startIndex, get<0>(lineRange), index - startIndex, get<1>(lineRange) -get<0>(lineRange)));
				foundWord = true;
			}
		}
	}

	int lastWhiteCol = image.rows;

	for (int i = image.cols - 1; i >= 0; i--)
		if (whitePixels[i] != 0)
		{
			lastWhiteCol = i;
			break;
		}

	if (startIndex < lastWhiteCol)
	{
		Mat x = image.colRange(startIndex, lastWhiteCol);
		lines.push_back(x);
		wordRects.push_back(Rect(startIndex, get<0>(lineRange), lastWhiteCol - startIndex, get<1>(lineRange) -get<0>(lineRange)));
	}

	m_words.push_back(wordRects);
	return lines;
}

vector<Mat> separate_words(vector<int> whiteSpaces, Mat line, tuple<int, int> lineRange)
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

	return find_words(whiteSpaces, line, spaceWeight, lineRange);
}
// Calculate the number of white pixels in the matrix
Mat calculateFeatureVector(Mat sample)
{
	Mat features = Mat(1, FEATURES_VECTOR_SIZE, CV_32FC1);

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
			features.at<float>(0, j * 4 + i) = whitePixels;
		}
	return features;
}

bool compareByX(const Rect &a, const Rect &b)
{
	return a.x < b.x;
}

vector<vector<vector<Mat>>> find_features(vector<vector<Mat>> words)
{
	vector<vector<vector<Mat>>> features;
	for (int i = 0; i < words.size(); i++)
	{
		vector<vector<Mat>> lineFeatures;
		vector<vector<Rect>> lineRects;
		for (int j = 0; j < words[i].size(); j++)
		{
			vector<Mat> wordFeatures;
			vector<Rect> wordRects;
			Mat sample = words[i][j];
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

			sort(boundingBox.begin(), boundingBox.end(), compareByX);

			for (int k = 0; k < contours.size(); k++)
			{
				Mat copy;
				sample.copyTo(copy);
				// Resize the contours
				resize(copy.colRange(boundingBox[k].x, boundingBox[k].x + boundingBox[k].width).rowRange(boundingBox[k].y
					, boundingBox[k].y + boundingBox[k].height), copy, boundingBox[k].size());
				int size = max(boundingBox[k].size().width, boundingBox[k].size().height);
				copyMakeBorder(copy, copy, 0, size - boundingBox[k].height, 0, size - boundingBox[k].width, BORDER_ISOLATED);
				resize(copy, copy, Size(32, 32));
				wordFeatures.push_back(calculateFeatureVector(copy));
				wordRects.push_back(Rect(m_words[i][j].x + boundingBox[k].x, m_words[i][j].y + boundingBox[k].y, boundingBox[k].width, boundingBox[k].height));
			}

			lineFeatures.push_back(wordFeatures);
			lineRects.push_back(wordRects);
		}
		features.push_back(lineFeatures); 
		m_letters.push_back(lineRects);
	}

	return features;
}

void interpret_results(Mat image, vector<vector<vector<int>>> results)
{
	Mat colorImage;
	cvtColor(image, colorImage, CV_GRAY2BGR);
	for (int i = 0; i < results.size(); i++)
		for (int j = 0; j < results[i].size(); j++)
		{
			rectangle(colorImage, m_words[i][j], Scalar(100, 125, 25));
			for (int k = 0; k < results[i][j].size(); k++)
				putText(colorImage, to_string(results[i][j][k]), Point(m_letters[i][j][k].x, m_letters[i][j][k].y + m_letters[i][j][k].height), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 20, 0), 2);
		}
	imshow("Text recognition", colorImage);
}

int main(int argc, char* argv[])
{
	string imgName;
	cout << "Choose the image to read\n";
	cin >> imgName;

	Mat image = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);
	Mat thresholdImage;
	threshold(image, thresholdImage, 200, 255, THRESH_BINARY_INV);
	vector<Mat> lines = find_lines(find_white_Pixels(thresholdImage, true), thresholdImage);
	vector<vector<Mat>> words;

	for (int i = 0; i < lines.size(); i++)
		words.push_back(separate_words(find_white_Pixels(lines[i], false), lines[i], m_lines[i]));

	vector<vector<vector<Mat>>> features = find_features(words);

	int model;
	cout << "Choose statistic model:\n" << "1. MLP\n" << "2. Bayes Classifier\n" << "3. R Trees\n" << "4. SVM\n";
	cin >> model;

	string fileName;
	vector<vector<vector<int>>> results;
	switch (model)
	{
	case MLP:
		fileName = "..\\MLPClassifier.yaml";
		results = mlp(fileName, features);
		break;
	case Bayes_Classifier:
		fileName = "..\\NormalBayesClassifier.yaml";
		results = bayes(fileName, features);
		break;
	case R_Trees:
		fileName = "..\\RTreesClassifier.yaml";
		results = r_trees(fileName, features);
		break;
	case Model::SVM:
		fileName = "..\\SVMClassifier.yaml";
		results = svm(fileName, features);
		break;
	default:
		break;
	}

	interpret_results(image, results);
	waitKey();
	return 0;
}