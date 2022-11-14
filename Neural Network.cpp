// Neural Network.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include "Net.h"
#include "TrainingData.h"

using namespace std;

void showVectorVals(string label, vector<double>& v);

int main()
{
	/*
	//instantiate our object
	//e.g {3,2,1}
	//first number is input layer, second number is hidden layer, third number is output layer
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net myNet(topology);

	vector<double> inputVals;
	//to train
	//feeds forward a bunch of input value
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	//we will pass the right answer aka targetVals
	//the right answers will then be propagated backwards
	myNet.backProp(targetVals);

	vector<double> resultVals;
	//after its been trained
	//we will pass in a container that will get the results
	myNet.getResults(resultVals);

	//we are going to loop this over and over to train it.

	//+Class Net
	//+feedForward()
	//+backProp()
	//+getResults
	*/

	
	//get our training data
	TrainingData trainData("trainingData.txt");

	// e.g., { 3, 2, 1 }
	vector<unsigned> topology;
	trainData.getTopology(topology);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;
	

	

	/*
	
	//uncomment this to generate your training data for the neural network it is training the neural network we made to be a XOR

	ofstream fout;
	string line;
	fout.open("trainingData.txt");


	while (fout)
	{

		//random training set for XOR --two inputs and one output
		cout << "topology: 2 4 1" << endl;
		line = "topology: 2 4 1\n";

		//2000 generated training data. if n1 and n2 are different than its true (XOR)
		for (int i = 2000; i >= 0; --i)
		{
			int n1 = (int)(2.0 * rand() / double(RAND_MAX));
			int n2 = (int)(2.0 * rand() / double(RAND_MAX));
			int t = n1 ^ n2; //should be 0 or 1
			line = line + "in: " + to_string(n1) + ".0 " + to_string(n2) + ".0\n";
			cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
			cout << "out: " << t << ".0" << endl;
			line = line + "out: " + to_string(t) + ".0\n";
		}


		//writes the string to the file
		fout << line << endl;
		break;
	}

	*/
	

	


	



};

void showVectorVals(string label, vector<double>& v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
