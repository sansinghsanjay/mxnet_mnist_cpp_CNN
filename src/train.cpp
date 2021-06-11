/*
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To train on MNIST color images dataset
Compile:
$ g++ train.cpp -lmxnet 
*/

// libraries
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <chrono>

// namespaces
using namespace std;
using namespace mxnet::cpp;

// global constants
/*
Number of instances in train data: 41580 (99% of the entire data)
Number of instances in validation data: 420 (1% of the entire data)
Total instances in the entire data: 42000
*/
const int BATCH_SIZE = 84;
const int IMAGE_W = 28;
const int IMAGE_H = 28;
const int IMAGE_C = 3;
const int NUM_CLASSES = 10;
const int EPOCHS = 20;
const float LEARNING_RATE = 0.01;

// function to round off double values up to 2 decimal digits
float round_fig(double d) {
	float r = (int) (d * 100 + 0.5);
	return (float) r / 100;
}

// CNN - architecture
Symbol SimpleCNN() {
	auto input_X = Symbol::Variable("data");
	auto input_Y = Symbol::Variable("label");
	// layer 1
	auto conv1 = Operator("Convolution")
		.SetParam("kernel", Shape(3,3))
		.SetParam("num_filter", 16)
		.SetInput("data", input_X)
		.CreateSymbol("conv1");
	auto relu1 = Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", conv1)
		.CreateSymbol("relu1");
	// layer 2
	auto conv2 = Operator("Convolution")
		.SetParam("kernel", Shape(3,3))
		.SetParam("num_filter", 32)
		.SetInput("data", relu1)
		.CreateSymbol("conv2");
	auto relu2 = Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", conv2)
		.CreateSymbol("relu2");
	// layer 3
	auto conv3 = Operator("Convolution")
		.SetParam("kernel", Shape(3,3))
		.SetParam("num_filter", 64)
		.SetInput("data", relu2)
		.CreateSymbol("conv3");
	auto relu3 = Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", conv3)
		.CreateSymbol("relu3");
	// dropout layer
	auto dropout1 = Operator("Dropout")
		.SetParam("p", 0.3)
		.SetInput("data", relu3)
		.CreateSymbol("dropout1");
	// flattening
	auto flatten = Operator("Flatten")
		.SetInput("data", dropout1)
		.CreateSymbol("flatten");
	// fully connected layer
	auto fc1 = Operator("FullyConnected")
		.SetParam("num_hidden", 64)
		.SetInput("data", flatten)
		.CreateSymbol("fc1");
	auto relu4 = Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc1)
		.CreateSymbol("relu4");
	// final layer - fully connected
	auto fc2 = Operator("FullyConnected")
		.SetParam("num_hidden", NUM_CLASSES)
		.SetInput("data", relu4)
		.CreateSymbol("fc2");
	auto softmax = Operator("SoftmaxOutput")
		.SetInput("data", fc2)
		.SetInput("label", input_Y)
		.CreateSymbol("softmax");
	// return softmax
	return softmax;
}

// MAIN FUNCTION
int main() {
	// variable declaration
	Context ctx = Context::cpu();
	map<string, NDArray> args_map;
	cout<<"Declared variables"<<endl;
	// initialising variables
	args_map["data"] = NDArray(Shape(BATCH_SIZE, IMAGE_C, IMAGE_W, IMAGE_H), ctx);
	args_map["label"] = NDArray(Shape(BATCH_SIZE), ctx);
	cout<<"Initialised variables"<<endl;
	// paths
	string trainDataCSVPath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/train_data.csv";
	string valDataCSVPath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/val_data.csv";
	string trainRecFilePath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/im2rec_files/mnistData_train.rec";
	string valRecFilePath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/im2rec_files/mnistData_val.rec";
	// declaring data iterator for train data
	auto trainImgIter = MXDataIter("ImageRecordIter")
		.SetParam("path_imgrec", trainRecFilePath)
		.SetParam("data_shape", Shape(IMAGE_C, IMAGE_W, IMAGE_H))
		.SetParam("batch_size", BATCH_SIZE)
		.CreateDataIter();
	cout<<"Declared training data iterator"<<endl;
	// delcaring data iterator for val data
	auto valImgIter = MXDataIter("ImageRecordIter")
		.SetParam("path_imgrec", valRecFilePath)
		.SetParam("data_shape", Shape(IMAGE_C, IMAGE_W, IMAGE_H))
		.SetParam("batch_size", BATCH_SIZE)
		.CreateDataIter();
	cout<<"Declared validation data iterator"<<endl;
	// initialising CNN
	auto cnn_net = SimpleCNN();
	cnn_net.InferArgsMap(ctx, &args_map, args_map);
	cout<<"Initialised CNN"<<endl;
	// optimization function
	Optimizer* optimizer = OptimizerRegistry::Find("adam");
	optimizer->SetParam("lr", LEARNING_RATE);
	cout<<"Successfully defined optimizer"<<endl;
	// preparing runtime
	auto *exec = cnn_net.SimpleBind(ctx, args_map);
	auto arg_names = cnn_net.ListArguments();
	// training model
	for(int epoch=0; epoch<EPOCHS; epoch++) {
		int batchCount = 0;
		trainImgIter.Reset();
		auto start_time = chrono::system_clock::now();
		while(trainImgIter.Next()) {
			auto batchData = trainImgIter.GetDataBatch();
			// copy batch data to args_map for training
			batchData.data.CopyTo(&args_map["data"]);
			batchData.label.CopyTo(&args_map["label"]);
			exec->Forward(true);
			exec->Backward();
			for(size_t i=0; i<arg_names.size(); i++) {
				if(arg_names[i] == "data" || arg_names[i] == "label")
					continue;
				optimizer->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
			batchCount += 1;
			if(batchCount % 5 == 0)
				cout<<"Epoch: "<<epoch + 1<<" Batch: "<<batchCount<<endl;
		}
		auto end_time = chrono::system_clock::now();
		float time_taken = round_fig(chrono::duration_cast<chrono::milliseconds> (end_time - start_time).count() / 1000.0);
		// validating model
		Accuracy accuracy;
		valImgIter.Reset();
		while(valImgIter.Next()) {
			auto batchData = valImgIter.GetDataBatch();
			batchData.data.CopyTo(&args_map["data"]);
			batchData.label.CopyTo(&args_map["label"]);
			exec->Forward(false);
			accuracy.Update(batchData.label, exec->outputs[0]);
		}
		cout<<"Epoch: "<<epoch + 1<<" | Training Time: "<<time_taken<<" secs | Val Acc: "<<accuracy.Get()<<endl;
	}
	delete exec;
	MXNotifyShutdown();
	return 0;
}
