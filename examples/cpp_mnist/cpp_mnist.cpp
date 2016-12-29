#include <caffe/caffe.hpp>
#include <memory>
#include "caffe/layers/memory_data_layer.hpp"
#include "endian.h"

static FILE *datafp = 0;
static FILE *labelfp = 0;
static unsigned int numOfImage = 0;
static unsigned int numOfRow = 0;
static unsigned int numOfColumn = 0;
static unsigned int labelCount = 0;


void openData()
{
    datafp = fopen("train-images-idx3-ubyte","rb");
    labelfp = fopen("train-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    numOfImage = 0;
    numOfRow = 0;
    numOfColumn = 0;

    unsigned int magicNumLabel = 0;
    labelCount = 0;

    fread(&magicNumLabel, sizeof(unsigned int), 1, labelfp);
    fread(&labelCount, sizeof(unsigned int ),1, labelfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelCount = be32toh(labelCount);

    fread(&magicNum, sizeof(unsigned int), 1, datafp);
    fread(&numOfImage, sizeof(unsigned int), 1, datafp);
    fread(&numOfRow, sizeof(unsigned int), 1, datafp);
    fread(&numOfColumn, sizeof(unsigned int), 1, datafp);

    magicNum = be32toh(magicNum);
    numOfImage = be32toh(numOfImage);
    numOfRow = be32toh(numOfRow);
    numOfColumn = be32toh(numOfColumn);

//    printf("magic number: %x\n", magicNum);
//    printf("num of image: %d\n", numOfImage);
//    printf("num of row: %d\n", numOfRow);
//    printf("num of column: %d\n", numOfColumn);
//    printf("magic number label: %x\n", magicNumLabel);
//    printf("num of label: %d\n", labelCount);   
}

void closeData()
{
    fclose(datafp);
    fclose(labelfp);
}

int main()
{
	/*float *data = new float[64*1*1*3*400];
	float *label = new float[64*1*1*1*400];


	for(int i = 0; i<64*1*1*400; ++i)
	{
		int a = rand() % 2;
        int b = rand() % 2;
        int c = a ^ b;
		data[i*2 + 0] = a;
		data[i*2 + 1] = b;
		label[i] = c;
	}*/

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());
   // caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());

  //  float testab[] = {0, 0, 0, 1, 1, 0, 1, 1};
  //  float testc[] = {0, 1, 1, 0};

  //  dataLayer_testnet_->Reset(testab, testc, 4);

    openData();

    float *data = new float[numOfImage * numOfRow * numOfColumn];

    for(int i = 0 ;i<numOfRow * numOfColumn*numOfImage;++i)
    {

        unsigned char pixel;

        fread(&pixel, sizeof(unsigned char), 1, datafp);

        data[i] = (float) pixel / 255.0f;
    }


    float *label  = new float[labelCount];

    for(int i =0;i<labelCount;++i)
    {
        unsigned char _label = 0;
        fread(&_label, sizeof(unsigned char), 1, labelfp);

        label[i] = _label;
    }






    dataLayer_trainnet->Reset(data, label, numOfImage);

    
    solver->Solve();

    closeData();

    return 0;
/*
    boost::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("./model.prototxt", caffe::TEST));
    //testnet->CopyTrainedLayersFrom("XOR_iter_5000000.caffemodel");



    testnet->ShareTrainedLayersWith(solver->net().get());

    caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());

    dataLayer_testnet->Reset(testab, testc, 4);

    testnet->Forward();

    boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("output");

    const float* begin = output_layer->cpu_data();
    const float* end = begin + 4;
    
    std::vector<float> result(begin, end);

    for(int i = 0; i< result.size(); ++i)
    {
    	printf("input: %d xor %d,  truth: %f result by nn: %f\n", (int)testab[i*2 + 0], (int)testab[i*2+1], testc[i], result[i]);
    }
*/
	return 0;
}
