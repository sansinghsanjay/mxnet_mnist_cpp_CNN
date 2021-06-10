/*
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To train on MNIST color images dataset
Compile:
$ 
*/

// libraries
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>

// namespaces
using namespace std;
using namespace mxnet::cpp;

// global constants
const int BATCH_SIZE = 100;
const int IMAGE_W = 28;
const int IMAGE_H = 28;
const int IMAGE_C = 3;

// MAIN FUNCTION
int main() {
	// variable declaration
	Context ctx = Context::cpu();
	auto trainDataX = NDArray(Shape(BATCH_SIZE, IMAGE_C, IMAGE_W, IMAGE_H), ctx);
	auto trainDataY = NDArray(Shape(BATCH_SIZE), ctx);
	auto valDataX = NDArray(Shape(BATCH_SIZE, IMAGE_C, IMAGE_W, IMAGE_H), ctx);
	auto valDataY = NDArray(Shape(BATCH_SIZE), ctx);
	// paths
	string trainDataCSVPath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/train_data.csv";
	string valDataCSVPath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/val_data.csv";
	string recFilePath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/im2rec_files/trainList.rec";
	// declaring csv iterator for train and validation csv files
	auto trainImgIter = MXDataIter("ImageRecordIter");
	trainImgIter.SetParam("path_imgrec", recFilePath);
	trainImgIter.SetParam("data_shape", Shape(IMAGE_C, IMAGE_H, IMAGE_W));
	trainImgIter.SetParam("batch_size", BATCH_SIZE);
	trainImgIter.CreateDataIter();
	// running data iterators
	int batchCount = 0;
	trainImgIter.Reset();
	while(trainImgIter.Next()) {
		auto dataBatch = trainImgIter.GetDataBatch();
		batchCount += 1;
		cout<<"Batch: "<<batchCount<<endl;
		//cout<<dataBatch.data<<endl;
		//cout<<dataBatch.label<<endl;
		//if(batchCount == 5)
		//	break;
	}
	// updating status
	cout<<"1. Declared trainData and valData..."<<endl;
	cout<<"2. Added paths of train and validation csv..."<<endl;
	cout<<"3. Created CSVIter..."<<endl;
	return 0;
}
