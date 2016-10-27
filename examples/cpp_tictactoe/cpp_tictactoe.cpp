#include <caffe/caffe.hpp>
#include <memory>
#include "caffe/layers/memory_data_layer.hpp"

static double rate = 0.1;
static unsigned int count = 0;

static bool isWin(int position[9], int side)
{
    if( (position[0] == side && position[1] == side && position[2] == side) ||
        (position[3] == side && position[4] == side && position[5] == side) ||
        (position[6] == side && position[7] == side && position[8] == side) ||
        (position[0] == side && position[3] == side && position[6] == side) ||
        (position[1] == side && position[4] == side && position[7] == side) ||
        (position[2] == side && position[5] == side && position[8] == side) ||
        (position[0] == side && position[4] == side && position[8] == side) ||
        (position[2] == side && position[4] == side && position[6] == side))
    {
        return true;
    }
    else
    {
        return false;
    }
}

float getReward(NeuralNetwork<double> &network, std::vector<double> &position, int side, int depth)
{
    std::vector<double> outputs;
    network.getResult(position, outputs);
    float q = outputs[0];
    return q;
}

void recursiveTrain(NeuralNetwork<double> &network, NeuralNetwork<double>::MiniBatch &miniBatch, int position[9], int side, int depth)
{
    for(int i = 0;i<9;++i)
    {
        //if ((depth == 1 && i == 4) || (depth == 3 && (i == 5 || i==3 || i==1 || i==7)) || (depth != 1 && depth !=3))
        if (position[i] == 0)
        {
            position[i] = side;
            float reward = 0.0;
            bool notset = true;
            bool hasWon = false;
            
            if (isWin(position,  side))
            {
                hasWon = true;
                if (side == 1)
                {
                    reward = 1.0; //std::min(1.0, 0.5 + 0.05 * (18-depth));
                }
                else
                {
                    reward = -1.0; //std::max(0.0, 0.5 - 0.05 * (18-depth));
                }
            }
            else
            {
                for(int k=0; k<9; ++k)
                {
                    if (position[k] !=0)
                    {
                        continue;
                    }
                    std::vector<double> inputs;

                    for(int e = 0;e<9;++e)
                    {
                        inputs.push_back(position[e]);
                    }
                    inputs[k] = -side;

                    float q = getReward(network, inputs, -side, depth + 1);

                    if (notset)
                    {
                        notset = false;
                        reward = q;
                    }
                    else if (side == 1)
                    {
                        if (q < reward)
                        {
                            reward = q;
                        }
                    }
                    else
                    {
                        if (q > reward)
                        {
                            reward = q;
                        }
                    }
                }
                reward = 0.9 * reward;
            }

            //reward = (reward + 2.0 ) * 0.5

            std::vector<double> labelVector;
            labelVector.push_back(reward);
            std::vector<double> inputVector;
            inputVector.push_back(position[0]);
            inputVector.push_back(position[1]);
            inputVector.push_back(position[2]);
            inputVector.push_back(position[3]);
            inputVector.push_back(position[4]);
            inputVector.push_back(position[5]);
            inputVector.push_back(position[6]);
            inputVector.push_back(position[7]);
            inputVector.push_back(position[8]);

            NeuralNetwork<double>::TrainingData d(inputVector, labelVector);

            miniBatch.push_back(d);

            if (!hasWon)
            {
                recursiveTrain(network, miniBatch, position, -side, depth +1);
            }
            position[i] = 0;
        }
    }
}

int main()
{
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

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());
    
    float testab[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float testc[] = {0, 1, 1, 0};

    dataLayer_trainnet->Reset(data, label, 25600);

    solver->Solve();

    std::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("./model.prototxt", caffe::TEST));
    testnet->CopyTrainedLayersFrom("XOR_iter_5000000.caffemodel");

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

	return 0;
}