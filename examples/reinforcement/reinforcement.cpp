#include <caffe/caffe.hpp>
#include <memory>
#include "caffe/layers/memory_data_layer.hpp"

int main()
{
	/*const std::string model = "/Users/shiyan/caffe/examples/reinforcement/train.prototxt";
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Net<float> *net = new caffe::Net<float>(model, caffe::TRAIN);
	//net->init();
	printf("haha %d\n", net->has_layer("data"));

	caffe::MemoryDataLayer<float> *dataLayer = (caffe::MemoryDataLayer<float> *) net->layer_by_name("data").get();
*/
	float *data = new float[64*1*1*3*400];
	float *label = new float[64*1*1*1*400];


	for(int i = 0; i<64*1*1*400; ++i)
	{
		int a = rand() % 2;
        int b = rand() % 2;
        int c = a ^ b;
		data[i*2 + 0] = a;
		data[i*2 + 1] = b;
		label[i] = c;
	}

//	dataLayer->Reset(data, label, 128);


    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("/home/shiy/caffe/examples/reinforcement/solver.prototxt", &solver_param);

    std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("fuck").get());
    //caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (solver->test_net()->layer_by_name("fuck").get());
    
    //printf("input blob size:%d\n", solver->net()->num_inputs());

    //dataLayer_testnet->Reset(data, label, 256);
    dataLayer_trainnet->Reset(data, label, 25600);

    solver->Solve();

	return 0;
}