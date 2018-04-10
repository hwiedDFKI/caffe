#include <caffe/caffe.hpp>
#include <memory>
#include <map>
#include "caffe/layers/memory_data_layer.hpp"

static double rate = 0.1;
static unsigned int count = 0;


// a helper function to tell if one side wins
// when side == 1 means black wins
// when side == -1 means white wins
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

// given a position, this function returns a reward. a number close to 1 means black will likely win
// a number close to -1 means white will likely win
void getReward(boost::shared_ptr<caffe::Net<float> > &testnet, std::vector<float> &position, std::vector<float> &label)
{
    //get the input layer of the neural network
    caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
    // assign the current position to the input of the neural network
    dataLayer_testnet->Reset(&position[0], &label[0], 5477);

    // do a forward pass
    testnet->Forward();

    //get the output
    boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("output");

    const float* q = output_layer->cpu_data();

    //copy the output to label. the output is a reward.
    memcpy(&label[0], q, sizeof(float) * 5477);
}

// given a position, this function generates a hash key for the position.
// the hash key is a 32bit integer. This purpose of this hash key is to 
// rull out duplicated position.
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

// this function generates all possible positions of tic tac toe
void preGeneratePermutations(unsigned int &id, std::vector<float> &data, 
        std::vector<float> &label, std::map<unsigned int, unsigned int> &hash, int position[0], int side, int depth)
{
    // travel through all 9 locations
    for(int i = 0; i < 9; ++i)
    {
        // if this location is empty
        if (position[i] == 0)
        {
            // occupy the current location with the current side.
            // 1 means black
            // -1 means white
            // the game always start with black

            position[i] = side;
            float reward = 0.0;
            bool notset = true;
            bool hasWon = false;
            
            // decide if the current side wins the game
            if (isWin(position,  side))
            {
                hasWon = true;
            }

            // generate hash key for the current position
            unsigned int key = keyForPosition(position);

            //printf("key:%d\n", key);

            // if the current location has not seen before.
            if (hash.find(key) == hash.end())
            {
                // save the current position in the hash map for training.
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

            // if no one has won, recursively generate the next position.
            if (!hasWon)
            {
                preGeneratePermutations(id, data, label, hash,position, -side, depth +1);
            }

            // recover the current location to "empty"
            position[i] = 0;
        }
    }
}

//given a position, this function tells you who should be the side to put a stone.
// basically this function counts the number of stones of each side and 
// figure out who should be the current side to put a stone
int getSide(int position[9])
{
    int posC = 0;
    for(int e = 0;e<9;++e)
    {
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

    return side;
}



int getBest(std::map<unsigned int, unsigned int> &hash, 
                    std::vector<float> &oldLabel, 
                    int position[9], int side)
{
    int bestId = -1;
    float reward = 0.0;
    bool notset = true;
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
            bestId = k;
        }
        else if (-side == 1)
        {
            if (q > reward)
            {
                reward = q;
                bestId = k;
            }
        }
        else
        {
            if (q < reward)
            {
                reward = q;
                bestId = k;
            }
        }
    }

    return bestId;
}



void displayPosition(FILE *fp, int position[9])
{
    for(int i = 0; i<9;++i)
    {
        if (i % 3 == 0)
        {
            printf("\n");
            fprintf(fp, "\n");
        }    

        if (position[i] == 1)
        {
            printf("O");
            fprintf(fp, "O");
        }
        else if(position[i] == -1)
        {
            printf("X");
            fprintf(fp, "X");
        }
        else if(position[i] == 0)
        {
            printf("_");
            fprintf(fp, "_");
        }
    }
}

void experiment(std::map<unsigned int, unsigned int> &hash, 
                    std::vector<float> &oldLabel)
{
    FILE *fp = fopen("test.txt", "w");
    int rightCount = 0;
    int allCount = 0;
    {
        int position[9] = {0};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {0, 0, 0,
                           0, 1, 0,
                           0, 0, 0};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    }

    {
        int position[9] = {0, 0, 0,
                           1, 1, -1,
                           0, 0, -1};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {0, 0, 1,
                           1, 1,  -1,
                           0, 0, -1};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    }

    {
        int position[9] = {0, 0, -1,
                           0, 1, 0,
                           1, 0, -1};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {0, 0, -1,
                           0, 1,  1,
                           1, 0, -1};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    }    

    {
        int position[9] = {0, 0, -1,
                           0, 0, 0,
                           1, 1, -1};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {0, 0, -1,
                           0, 0,  1,
                           1, 1, -1};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    }  

    {
        int position[9] = {0, 0, -1,
                           0, 1, 0,
                           0, 1, -1};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {0, 1, -1,
                           0, 1,  0,
                           0, 1, -1};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    }  

    {
        int position[9] = {-1, -1, 0,
                           0, 1, 0,
                           0, 1, 0};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {-1, -1, 1,
                           0, 1,  0,
                           0, 1, 0};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    } 

    {
        int position[9] = {0, 0, -1,
                           0, 1, 1,
                           -1, 1, -1};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected_1[9] = {0, 0, -1,
                           1, 1,  1,
                           -1, 1, -1};
        int expected_2[9] = {0, 1, -1,
                           0, 1,  1,
                           -1, 1, -1};
        bool result1 = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected_1[i])
            {
                result1 = false;
            }
        }

        bool result2 = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected_2[i])
            {
                result2 = false;
            }
        }

        printf("\n--- ---- ---:%s\n\n", (result1||result2)?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", (result1||result2)?"right":"wrong");
        if((result1||result2) == true)
        {
            rightCount++;
        }
        allCount++;
    } 

    {
        int position[9] = {1, -1, -1,
                           0, 1,  0,
                           -1, 1, 0};
        int side = getSide(position);
        int bestId = getBest(hash, oldLabel, position, side);

        printf("--- Test ---:");
        fprintf(fp, "--- Test ---:");
        displayPosition(fp, position);
        position[bestId] = -side;
        printf("\n");
        fprintf(fp, "\n");
        displayPosition(fp, position);
        int expected[9] = {1, -1, -1,
                           0, 1,  0,
                           -1, 1, 1};
        bool result = true;
        for(int i=0;i<9;++i)
        {
            if (position[i] != expected[i])
            {
                result = false;
            }
        }
        printf("\n--- ---- ---:%s\n\n", result?"right":"wrong");
        fprintf(fp, "\n--- --- ---:%s\n\n", result?"right":"wrong");
        if(result == true)
        {
            rightCount++;
        }
        allCount++;
    } 
    printf("\nOverall result:%d / %d\n", rightCount, allCount);
    fprintf(fp, "\nOverall result:%d / %d\n", rightCount, allCount);
    fclose(fp);
}


// this function generates training data for one epoch
// hash is a tabel mapping from the key of a position to the index of the position in "data"
// there are 5477 positions in total (only unique positions are countered)
// each training data is formed by 1 position and 1 label
// 1 position is formed by 9 locaitons. if a location is 1, means a black stone,
// if a location is 0, means empty
// if a location is -1, means a white stone.
// a label is a reward number. 1 means black wins, -1 means white wins.
// a number closer to 1 means black is likely to win.
// a number closer to -1 means white is likely to win.
void getTrainingData(std::map<unsigned int, unsigned int> &hash, 
                    std::vector<float> &oldLabel, 
                    std::vector<float> &data, std::vector<float> &label)
{
    //start with an empty board, all 9 locations are empty
    int position[9] = {0};

    //overall, there are 5477 positions
    //loop for all of them
    for(int i = 5477-1; i>=0; i--)
    {
        float reward = 0.0;
        bool notset = true;
        bool hasWon = false;

        int posC = 0;

        // this for loop counts the number of stones on each side, 
        // and figure out who should the be current side to lay a stone
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

        // if black is the current side, side == 1
        // if white is the current side, side == -1
            
        // if the black wins the game with the current position
        if (isWin(position,  1))
        {
            // set reward to 1
            hasWon = true;
            reward = 1.0; //std::min(1.0, 0.5 + 0.05 * (18-depth));
        }
        else if(isWin(position, -1))
        {
            // if the white wins the game with the current position
            // set the reward to -1
            hasWon = true;
            reward = -1.0; //std::max(0.0, 0.5 - 0.05 * (18-depth));
        }
        else
        {
            // if no one wins the position
            int count = 0;
            // loop over all locations on the board
            for(int k=0; k<9; ++k)
            {
                // if the current location is not empty, skip the current location
                if (position[k] !=0)
                {
                    continue;
                }
    
                // if the current location is empty,
                // first, copy the current position into a buffer inputs
                int inputs[9];
                memcpy(inputs, position, sizeof(int) * 9);

                //put a stone of the enemy side at the current location
                inputs[k] = -side;

                // generate the hash key for the current position (with the enemy stone)
                unsigned int inputKey = keyForPosition(&inputs[0]);
                unsigned int id = hash[inputKey];
                // get the probability of enemy winning the game
                float q = oldLabel[id];

                // if current side == 1 (black), find the smallest reward, (white win)
                // if current side == -1 (white), find the largest reward, (black win)
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

                //reward += q;
                //count ++;
            }
            //if (count)
            {
                // so far, the reward is for the next step
                // here we multiply the reward of the next step with a decay number 0.7 (should be 0.9 for DQN)
                reward = 0.7 * reward; // count;
            }
        }

        // assign the reward to the current position.
        label[i] = reward;
    }
}

void testResult()
{
    std::vector<float> data;
    std::vector<float> label;
    std::vector<float> oldLabel;
    std::map<unsigned int, unsigned int> hash;
    unsigned int id = 0;
    int position[9] = {0};

    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    label.push_back(0);
    hash[0] = 0;
    preGeneratePermutations(id, data, label, hash, position, 1, 1);
    oldLabel.resize(label.size(), 0);

    boost::shared_ptr<caffe::Net<float> > testnet;

    testnet.reset(new caffe::Net<float>("./model.prototxt", caffe::TEST));

    testnet->CopyTrainedLayersFrom("./TicTacToe_iter_995000.caffemodel");
    getReward(testnet, data, oldLabel);
           
    experiment(hash, oldLabel);
}

int main()
{
    //testResult();
    //return 0;

#ifdef CPU_ONLY
   caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
   caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());

    boost::shared_ptr<caffe::Net<float> > testnet;

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


    for(int i = 0; i < 10000000; ++i)
    {
        getReward(testnet, data, oldLabel);

        if (i%5000 == 0)
        {
            experiment(hash, oldLabel);
        }

        getTrainingData(hash, oldLabel, data, label);

        dataLayer_trainnet->Reset(&data[0], &label[0], 5477);

        solver->Step(1);
    }

    getReward(testnet, data, oldLabel);
    experiment(hash, oldLabel);

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
