#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iterator>

using namespace std;

int main(int argv, char **argc){
    string inputFileName = "examples/ner.txtO", outputFileName = "pos_out.txt";
    string modelFile = "models/model.ritter_ptb_alldata_fixed.20130723";
    string jarFile = "ark-tweet-nlp-0.3.2.jar";
    string cmdLine, fileTemp = "#temppos.txt";

    fstream arqIn, arqOut;
    string word, pos, conf;

    char line[256];

    //this part of the converter is written for windows and have to be adapted for other systems

    if (argv>1)
        inputFileName = argc[1];
    else{
        cout << "USAGE: " << argc[0] << " inputFile outputFile posModelFile jarTaggerFile" << endl;
        cout << "inputFile: file with a list of tweets perline, mandatory argument" << endl;
        cout << "outputFile: tokenized outputfile with words and POS tags" << endl;
        cout << "posModelFile: file with the POS tagset" << endl;
        cout << "jarTaggerFile: java compiled file of the tagger" << endl;
        //return 1;
    }
    if (argv>2)
        outputFileName = argc[2];
    if (argv>3)
        modelFile = argc[3];
    if (argv>4)
        jarFile = argc[4];

    //removes temporary outputfile
    cmdLine = "del \"" + fileTemp + "\" >nul 2>nul";
    system(cmdLine.c_str());

    //run tweetnlp pos tagger
    cmdLine = "java -XX:ParallelGCThreads=2 -Xmx500m -jar " + jarFile +
              " --output-format conll --model " + modelFile + " ";
    cmdLine += inputFileName + " >> " + fileTemp;
    system(cmdLine.c_str());

    //end of SO dependent part

    //opens output file
    arqOut.open(outputFileName.c_str(), fstream::out);

    //opens input file
    arqIn.open(fileTemp.c_str(), fstream::in);
    if (!arqIn.is_open()){
        cout << "Invalid Input File" << endl;
        return 2;
    }

    //starts reading process
    while (!arqIn.eof()){
        //reads a tweet line
        arqIn.getline(line,255);

        //verifies if it is a valid line
        if (strlen(line) > 2){
            stringstream sline;
            sline << line;

            //breaks line
            sline >> word >> pos >> conf;

            //outputs word and POS, confidence is discarded
            arqOut << word << "\t" << pos << endl;
        }
        else{
            //outputs a break if no triple is present, new tweet
            arqOut << endl;
        }
    }

    //closes files
    arqIn.close();
    arqOut.close();
}
