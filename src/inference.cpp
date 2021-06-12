/*
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To make inference after loading MXNet model trained in C++
Compile:
$ g++ inference.cpp -lmxnet
*/

// libraries
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>

// namespaces
using namespace std;
using namespace mxnet::cpp;

// global constants
const int BATCH_SIZE = 20;
const int IMAGE_W = 28;
const int IMAGE_H = 28;
const int IMAGE_C = 3;
const int NUM_CLASSES = 10;
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
		.SetParam("ignore_label", -1)
		.SetParam("multi_output", false)
		.SetParam("use_ignore", false)
		.SetParam("normalization", "null")
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
	// paths
	string valRecFilePath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/dataset/im2rec_sampleSet/mnistSampleSet_val.rec";
	string trainedModelPath = "/home/sansingh/github_repo/mxnet_mnist_cpp_CNN/trained_model/model";
	// delcaring data iterator for val data
	auto valImgIter = MXDataIter("ImageRecordIter")
		.SetParam("path_imgrec", valRecFilePath)
		.SetParam("data_shape", Shape(IMAGE_C, IMAGE_W, IMAGE_H))
		.SetParam("batch_size", BATCH_SIZE)
		.CreateDataIter();
	cout<<"Declared validation data iterator"<<endl;
	// initialising variables
	args_map["data"] = NDArray(Shape(BATCH_SIZE, IMAGE_C, IMAGE_W, IMAGE_H), ctx);
	args_map["label"] = NDArray(Shape(BATCH_SIZE), ctx);
	cout<<"Initialised variables"<<endl;
	// load model parameters - weights and bias
	NDArray::Load(trainedModelPath, nullptr, &args_map);
	cout<<"Successfully loaded trained model"<<endl;
	// initialising CNN
	auto cnn_net = SimpleCNN();
	cnn_net.InferArgsMap(ctx, &args_map, args_map);
	cout<<"Initialised CNN"<<endl;
	// optimization function
	Optimizer* optimizer = OptimizerRegistry::Find("adagrad");
	optimizer->SetParam("lr", LEARNING_RATE);
	cout<<"Successfully defined optimizer"<<endl;
	// preparing runtime
	auto *exec = cnn_net.SimpleBind(ctx, args_map);
	auto arg_names = cnn_net.ListArguments();
	args_map = exec->arg_dict();
	cout<<"Successfully created runtime"<<endl;
	// validating model on validation data
	Accuracy val_accuracy;
	valImgIter.Reset();
	float avg_valAcc = 0.0f;
	int batchCount = 0;
	while(valImgIter.Next()) {
		auto batchData = valImgIter.GetDataBatch();
		batchData.data.CopyTo(&args_map["data"]);
		batchData.label.CopyTo(&args_map["label"]);
		exec->Forward(false);
		val_accuracy.Update(batchData.label, exec->outputs[0]);
		avg_valAcc += val_accuracy.Get();
		batchCount++;
	}
	avg_valAcc = round_fig(avg_valAcc / batchCount);
	cout<<"Obtained Accuracy: "<<avg_valAcc<<endl;
	delete exec;
	MXNotifyShutdown();
	return 0;
}
