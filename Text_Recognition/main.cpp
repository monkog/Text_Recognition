#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

enum Model
{
	MLP = 1
	, Bayes_Classifier = 2
	, R_Trees = 3
	, SVM = 4
};

void mlp(Mat image, int width, int cols, int rows)
{
	CvANN_MLP mlp;
	Mat output;
	mlp.create(image);
	mlp.train(image, output, Mat());
}

void parse_training_data(string filename, int model, int width, int cols, int rows)
{
	Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	switch (model)
	{
	case MLP:
		mlp(image, width, cols, rows);
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
	int model;
	cout << "Choose statistic model:\n" << "1. MLP\n" << "2. Bayes Classifier\n" << "3. R Trees\n" << "4. SVM\n";
	cin >> model;

	int width = 20, cols = 100;
	//int width = 40, cols = 40;

	parse_training_data(filename, model, width, cols, digits ? 10 : 26);
}

void choose_training_file(bool digits)
{
	string filename;
	cout << "Type filename\n";
	cin >> filename;

	string lettersFile = "..\letters.png";
	string digitsFile = "..\digits.png";

	choose_clasifier(digits, lettersFile);
}

int main(int argc, char* argv[])
{
	int result;
	cout << "Choose training set:\n" << "1. Letters\n" << "2. Digits\n";
	cin >> result;

	bool isDigit = result == 2;
	choose_training_file(isDigit);

	return 0;
}