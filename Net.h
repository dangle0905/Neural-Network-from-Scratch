#pragma once
#include <vector>
#include "Neuron.h"
#include <cassert>

using namespace std;

class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError;}

private:
	//defining our Data Type Layer
	typedef vector<Neuron> Layer;
	vector<Layer> m_layers; //m_layers[LayerNum][nueronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;

};
