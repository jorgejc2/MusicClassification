#include "wavread.h"

int main(int argc, char* argv[])
{
    /* take a string argument to a relative file path to open */
    const char* filePath;
    string input;
    if (argc <= 1)
    {
        cout << "Input wave file name: ";
        cin >> input;
        cin.get();
        filePath = input.c_str();
    }
    else
    {
        filePath = argv[1];
        cout << "Input wave file name: " << filePath << endl;
    }

    /* create a wav file reader object to */
    wavFileReader wav_obj;

    /* get the samples from the wav file */
    int8_t* wav_samples;
    int num_samples = wav_obj.readFile(&wav_samples, filePath, 1);

    /* readFile was not successful */
    if (num_samples < 0) 
        return 1;
    
    /* typecast to int16_t since the files being worked with use 16 bits per sample */
    int16_t* wav_samples_16 = (int16_t *)wav_samples;

    /* save results to a file */
    // FILE *fp;
    // fp = fopen("../../OutputText/c_out.txt", "w");

    // for (int i = 0; i < num_samples/2; i++) {
    // fprintf(fp, "%d \n", (int)wav_samples_16[i]);
    // }

    // fclose(fp);

    /* free pointer holding the wav sample data */
    delete [] wav_samples;

    return 0;
}