#include <caffe/caffe.hpp>
#include <memory>
#include <map>
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

void getReward(std::shared_ptr<caffe::Net<float> > &testnet, std::vector<float> &position, std::vector<float> &label)
{
    caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
    dataLayer_testnet->Reset(&position[0], &label[0], 5477);

    testnet->Forward();

    boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("output");

    const float* q = output_layer->cpu_data();

    memcpy(&label[0], q, sizeof(float) * 5477);
}

unsigned int keyForPosition(int position[9])
{
    unsigned int result = 0;

    result |= (position[0] + 1) << 16;
    result |= (position[1] + 1) << 14;
    result |= (position[2] + 1) << 12;
    result |= (position[3] + 1) << 10;
    result |= (position[4] + 1) << 8;
    result |= (position[5] + 1) << 6;
    result |= (position[6] + 1) << 4;
    result |= (position[7] + 1) << 2;
    result |= (position[8] + 1) << 0;

    return result;
}

void preGeneratePermutations(unsigned int &id, std::vector<float> &data, std::vector<float> &label, std::map<unsigned int, unsigned int> &hash, int position[0], int side, int depth)
{
    for(int i = 0; i < 9; ++i)
    {
        if (position[i] == 0)
        {
            position[i] = side;
            float reward = 0.0;
            bool notset = true;
            bool hasWon = false;
            
            if (isWin(position,  side))
            {
                hasWon = true;
            }

            unsigned int key = keyForPosition(position);

            //printf("key:%d\n", key);

            if (hash.find(key) == hash.end())
            {
                hash[key] = id;
                label.push_back(reward);
                data.push_back(position[0]);
                data.push_back(position[1]);
                data.push_back(position[2]);
                data.push_back(position[3]);
                data.push_back(position[4]);
                data.push_back(position[5]);
                data.push_back(position[6]);
                data.push_back(position[7]);
                data.push_back(position[8]);
                id ++;
            }

            if (!hasWon)
            {
                preGeneratePermutations(id, data, label, hash,position, -side, depth +1);
            }

            position[i] = 0;
        }
    }
}


void getTrainingData(std::map<unsigned int, unsigned int> &hash, 
                    std::vector<float> &oldLabel, 
                    std::vector<float> &data, std::vector<float> &label)
{
    int position[9] = {0};
    for(int i = 5477-1; i>=0; i--)
    {
        float reward = 0.0;
        bool notset = true;
        bool hasWon = false;

        int posC = 0;

        for(int e = 0;e<9;++e)
        {

            position[e] = data[i*9 + e];
            if (position[e] == 1)
            {
                posC ++;
            }
            else if (position[e] == -1)
            {
                posC --;
            }
        }

        int side = -1;

        if (posC > 0)
            side = 1;
            
        if (isWin(position,  1))
        {
            hasWon = true;
            reward = 1.0; //std::min(1.0, 0.5 + 0.05 * (18-depth));
        }
        else if(isWin(position, -1))
        {
            hasWon = true;
            reward = -1.0; //std::max(0.0, 0.5 - 0.05 * (18-depth));
        }
        else
        {
            for(int k=0; k<9; ++k)
            {
                if (position[k] !=0)
                {
                    continue;
                }
    
                int inputs[9];
                memcpy(inputs, position, sizeof(int) * 9);

                inputs[k] = -side;

                unsigned int inputKey = keyForPosition(&inputs[0]);
                unsigned int id = hash[inputKey];
                float q = oldLabel[id];

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

        label[i] = reward;
    }
}

int main()
{
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());

    std::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("./model.prototxt", caffe::TEST));
    testnet->ShareTrainedLayersWith(solver->net().get());

    std::vector<float> data;
    std::vector<float> label;
    std::vector<float> oldLabel;
    std::map<unsigned int, unsigned int> hash;
    unsigned int id = 0;
    int position[9] = {0};

    preGeneratePermutations(id, data, label, hash, position, 1, 1);
    oldLabel.resize(label.size(), 0);

    printf("data size: %d label size: %d hash size: %d\n", data.size(), label.size(), hash.size());


    for(int i = 0; i < 5000000; ++i)
    {

        getReward(testnet, data, oldLabel);

        getTrainingData(hash, oldLabel, data, label);

        dataLayer_trainnet->Reset(&data[0], &label[0], 5477);

        solver->Step(1);
    }
 //   float testab[] = {0, 0, 0, 1, 1, 0, 1, 1};
 //   float testc[] = {0, 1, 1, 0};

    //dataLayer_trainnet->Reset(data, label, 25600);

    //solver->Solve();
/*
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
    */

	return 0;
}