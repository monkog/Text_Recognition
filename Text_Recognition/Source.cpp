//
///////////////////////////////////////////////////////////////////////
//#include "ml.h"
//#include "cxcore.h"
//#include "cv.h"
//#include "stdio.h"
//
////using OpenCV CvANN_MLP
//void InitializeRandoms()
//{
//	srand(4711);
//}
//double RandomDouble(double low, double high)
//{
//	return ((double)rand() / RAND_MAX*(high - low) + low);
//}
//int RandomInt(int low, int high)
//{
//	return rand() % (high - low + 1) + low;
//}
//
//int main()
//{
//	//input
//	// (0.2 0.8 0.1) (0.9,0.2,0.3) (0.6,0.6,0.6)
//	// classify
//	// (1 0 0) ( 0 1 0) ( 0 0 1)
//
//	float in[] = {
//		0.2, 0.8, 0.1,
//		0.9, 0.2, 0.3,
//		0.6, 0.6, 0.6 };
//	float out[] = {
//		1, 0, 0,
//		0, 1, 0,
//		0, 0, 1 };
//	int layer[] = { 3, 6, 3 };
//
//	CvMat *input = cvCreateMat(3, 3, CV_32FC1);
//	CvMat *target = cvCreateMat(3, 3, CV_32FC1);
//	CvMat *layersize = cvCreateMat(1, 3, CV_32SC1);;
//
//
//	cvInitMatHeader(input, 3, 3, CV_32FC1, in);
//	cvInitMatHeader(target, 3, 3, CV_32FC1, out);
//	cvInitMatHeader(layersize, 1, 3, CV_32SC1, layer);
//
//	CvANN_MLP bpn(layersize, CvANN_MLP::SIGMOID_SYM, 1, 1);
//
//	int iter = bpn.train(input, target, NULL, 0,
//		CvANN_MLP_TrainParams(cvTermCriteria
//		(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5000, 0.001),
//		CvANN_MLP_TrainParams::BACKPROP,
//		0.1, 0.1)
//		);
//
//	printf("Train Network: %d iterations\n", iter);
//
//	float eva[30];
//
//	int i, j, k = 0;
//	float err = 0;
//	for (i = 0; i<9; ++i)
//		eva[i] = in[i];
//
//	InitializeRandoms();
//
//	for (i = 3; i<10; ++i)
//	{
//		k = RandomInt(0, 2);
//		for (j = 0; j<3; ++j)
//		{
//			err = RandomDouble(-0.1, 0.1);
//			eva[i * 3 + j] = in[k * 3 + j] + err;
//		}
//	}
//
//	CvMat *ineva = cvCreateMat(10, 3, CV_32FC1);;
//	CvMat *output = cvCreateMat(10, 3, CV_32FC1);
//	cvInitMatHeader(ineva, 10, 3, CV_32FC1, eva);
//
//	bpn.predict(ineva, output);
//
//	printf("\tIn Classify\n\n");
//	fprintf(f, "\tIn Classify\n\n");
//	for (i = 0; i<10; i++)
//
//	{
//		printf("(%.2f,%.2f,%.2f)
//			(%.3f, %.3f, %.3f)\n",
//			CV_MAT_ELEM(*ineva, float, i, 0),
//			CV_MAT_ELEM(*ineva, float, i, 1),
//			CV_MAT_ELEM(*ineva, float, i, 2),
//			CV_MAT_ELEM(*output, float, i, 0),
//			CV_MAT_ELEM(*output, float, i, 1),
//			CV_MAT_ELEM(*output, float, i, 2)
//			);
//
//	}
//	return 0;
//}
//
//OutPut:
//Train Network : 69 iterations
//In Classify
//
//(0.20, 0.80, 0.10) (0.959, 0.063, 0.058)
//(0.90, 0.20, 0.30) (0.050, 0.937, 0.073)
//(0.60, 0.60, 0.60) (0.036, 0.075, 0.937)
//(0.60, 0.69, 0.62) (0.036, 0.069, 0.942)
//(0.28, 0.85, 0.03) (0.944, 0.320, -0.010)
//(0.81, 0.28, 0.24) (0.107, 0.918, 0.054)
//(0.27, 0.73, 0.03) (0.944, 0.319, -0.010)
//(0.25, 0.76, 0.11) (0.939, 0.136, 0.020)
//(0.23, 0.76, 0.05) (0.947, 0.224, -0.002)
//(0.99, 0.18, 0.35) (0.030, 0.946, 0.085)
//
///////////////////////////////////////////////////////////////////////
//
//I hope this useful to some body like me.