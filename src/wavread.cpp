#include "wavreader/wavread.h"

wavFileReader::wavFileReader() {}

wavFileReader::~wavFileReader() {};

// find the file size
/*
    Description: Get the size of the .wav file in bytes
    Inputs:
        FILE* fileObj -- .wav file object
    Outputs:
        None
    Returns:
        int -- size of .wav file in bytes
    Effects:
        None
*/
int wavFileReader::_getFileSize(FILE* fileObj)
{
    int fileSize = 0;
    /* sets file position indicator to the end of the file */
    fseek(fileObj, 0, SEEK_END);

    /* get number of bytes the file position indicator is from the beginning of the file */
    fileSize = ftell(fileObj);

    /* sets the file position indicator back to the start of the file */
    fseek(fileObj, 0, SEEK_SET);

    return fileSize;
}

wav_hdr wavFileReader::getWavHdr(const char* fileIn) {
    /* Check if .wav file can be opened */
    wav_hdr wavHeader;
    if (fileIn == nullptr) {
        fprintf(stderr, "Please set the current wav file to read from");
        return wavHeader;
    }
    FILE* wavFile = fopen(fileIn, "r");
    if (wavFile == nullptr)
    {
        fprintf(stderr, "Unable to open wave file: %s\n", fileIn);
        return wavHeader;
    }

    int headerSize = sizeof(wav_hdr), filelength = _getFileSize(wavFile), samplelength = filelength - headerSize;
    size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);

    return wavHeader;
}

/*
    Description: Takes a .wav file name and returns an array of its samples
    Inputs:
        const char* fileIn -- string containing the relative path and name of the file
        int display -- if set, prints information about the .wav file
    Outputs:
        int8_t** wav_buffer -- array holding the array of samples from the .wav file
    Returns:
        int -- -1 if operation was unsuccessful, else size of wav_buffer
    Effects:
        Allocates memory towards wav_buffer which caller must eventually deallocate
*/
int wavFileReader::readFile(int8_t **wav_buffer, const char* fileIn, int display = 1) {

    /* Check if .wav file can be opened */
    if (fileIn == nullptr) {
        fprintf(stderr, "Please set the current wav file to read from");
        return -1;
    }
    FILE* wavFile = fopen(fileIn, "r");
    if (wavFile == nullptr)
    {
        fprintf(stderr, "Unable to open wave file: %s\n", fileIn);
        return -1;
    }

    /* Read the .wav header */
    wav_hdr wavHeader;
    int headerSize = sizeof(wav_hdr), filelength = _getFileSize(wavFile), samplelength = filelength - headerSize;
    size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
    size_t totalBytesRead = 0;
    cout << "Header Read " << bytesRead << " bytes." << endl;
    cout << "Samplelength " << samplelength << endl;


    if (bytesRead > 0)
    {
        /* Read the data */
        uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
        uint64_t numSamples = wavHeader.ChunkSize / bytesPerSample; //How many samples are in the wav file?
        static const uint16_t BUFFER_SIZE = 4096; // read 4096 bytes at a time
        *wav_buffer = new int8_t[samplelength];
        int8_t *temp_buffer = *wav_buffer;
        if (wav_buffer == nullptr) {
            fprintf(stderr, "Bad allocation for wav_buffer\n");
            return -1;
        }

        int curr_offset = 0;
        while ((bytesRead = fread(&temp_buffer[curr_offset], 1, BUFFER_SIZE, wavFile)) > 0)
        {
            /** DO SOMETHING WITH THE WAVE DATA HERE **/
            totalBytesRead += bytesRead;
            curr_offset += BUFFER_SIZE;
        }
        // cout << "bytesPerSample " << bytesPerSample << endl;
        // cout << "totalBytesRead " << totalBytesRead << endl;

        /* print information about the wav file if requested */
        if (display) {

            cout << "File is                    :" << filelength << " bytes." << endl;
            cout << "Num of samples are         :" << samplelength << " bytes." <<endl;
            cout << "RIFF header                :" << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2] << wavHeader.RIFF[3] << endl;
            cout << "WAVE header                :" << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2] << wavHeader.WAVE[3] << endl;
            cout << "FMT                        :" << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2] << wavHeader.fmt[3] << endl;
            cout << "Data size                  :" << wavHeader.ChunkSize << endl;

            // Display the sampling Rate from the header
            cout << "Sampling Rate              :" << wavHeader.SamplesPerSec << endl;
            cout << "Number of bits used        :" << wavHeader.bitsPerSample << endl;
            cout << "Number of channels         :" << wavHeader.NumOfChan << endl;
            cout << "Number of bytes per second :" << wavHeader.bytesPerSec << endl;
            cout << "Data length                :" << wavHeader.Subchunk2Size << endl;
            cout << "Audio Format               :" << wavHeader.AudioFormat << endl;
            // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM

            cout << "Block align                :" << wavHeader.blockAlign << endl;
            cout << "Data string                :" << wavHeader.Subchunk2ID[0] << wavHeader.Subchunk2ID[1] << wavHeader.Subchunk2ID[2] << wavHeader.Subchunk2ID[3] << endl;
        }
    }
    fclose(wavFile);
    return samplelength;
}
