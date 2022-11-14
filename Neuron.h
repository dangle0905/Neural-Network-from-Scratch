#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal;}
	void calcOutputGradients(double targetVal);
	typedef vector<Neuron> Layer;
	void feedForward(const Layer & prevLayer);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);

private:
	static double eta; // [0.0 .. 1.0] overall net training rate
	static double alpha; // [0.0 .. 1.0] overall multiplier of last weight change (momentum)
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);

	//doesnt take a parameter and returns a number between 0 and 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	unsigned m_myIndex;
	vector<Connection> m_outputWeights;
	double m_gradient;
	typedef vector<Neuron> Layer;
	//it wont modify the object so its whole function is const
	double sumDOW(const Layer& nextLayer) const; 


};

