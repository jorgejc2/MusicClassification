#include "wavreader/wavread.h"
#include <math.h>
#include <vector>
#include <NumCpp.hpp>
#include <boost>
using namespace std;

int main(int argc, char* argv[])
{
    // const char* filePath;
    // string input;
    // if (argc <= 1)
    // {
    //     cout << "Input wave file name: ";
    //     cin >> input;
    //     cin.get();
    //     filePath = input.c_str();
    // }
    // else
    // {
    //     filePath = argv[1];
    //     cout << "Input wave file name: " << filePath << endl;
    // }

    int sample_rate = 4000;
    int length_ts_sec = 3;
    int length_ts1_sec = 1;
    int length_ts2_sec = 3;
    int total_ts_length = length_ts_sec + length_ts1_sec + length_ts2_sec;
    int ts_size = sample_rate * 7;

    int freq1 = 697;
    int freq2 = 1209;
    int freq3 = 1336;
    float freq1_rate = M_PI * 2 * freq1 / sample_rate;
    float freq2_rate = M_PI * 2 * freq2 / sample_rate;
    float freq3_rate = M_PI * 2 * freq3 / sample_rate;


    vector<float>ts(ts_size);

    for (int i = 0; i < ts.size(); i++) {
        int x = i % sample_rate;
        if (i < sample_rate * 3) {
            ts[i] = sin(x * freq1_rate) + sin(x * freq2_rate);
        }
        else if (i < sample_rate * 4) {
            ts[i] = 0.0;
        }
        else {
            ts[i] = sin(x * freq1_rate) + sin(x * freq3_rate);
        }
    }

    for (int i = 0; i < 100; i++)
        printf("%.15f ", ts[i]);


    return 0;
}
