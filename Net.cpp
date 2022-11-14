#include "Net.h"
#include "Neuron.h"
#include <iostream>

using namespace std;

double Net::m_recentAverageSmoothingFactor = 100.0;

Net::Net(const vector<unsigned>& topology)
{
	//defing our Data Type Layer
	typedef vector<Neuron> Layer;

	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		//create a new layer and append that to our m_layer container
		m_layers.push_back(Layer()); 

		//if layer number is the output layer which is the highest layer -1 then the number output is 0. 
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0: topology[layerNum + 1];
		
		//new layer is made now fill it with ith neurons, and adda bias neuron to the layer:
		//we do <= too to add the bias
		cout << "Layer: " << layerNum << ": " << endl;
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			//get the last element in the container
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			if (neuronNum == topology[layerNum]) 
			{
				cout << "Bias Neuron." << endl;;
			}
			else 
			{
				cout << "Made a Neuron." << endl;
			}
		
			//Force the bias nodes output val to 1.0. It's the last neuron created above
			m_layers.back().back().setOutputVal(1.0);
			
		}
	}
}

void Net::feedForward(const vector<double>& inputVals)
{
	//test for condtion
	assert(inputVals.size() == m_layers[0].size() - 1);

	//assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//forard propagation, looping thru each layer and then loop thru each neuron in the layer to go forward
	//we dont start at the input layer because its already set we start at the hidden layers
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		//point to prevLayer
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() -1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}

}

void Net::backProp(const vector<double>& targetVals)
{
	//calculate overal net error (RMS of output neruron errors)
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; //get average error squared
	m_error = sqrt(m_error); //RMS

	//Implement a recent average measurement 
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//calculate gradients on hidden layer
	for (unsigned layerNum = m_layers.size() - 2; layerNum >0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//for all layers from output to first hidden layer 
	//update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}

	}
}

void Net::getResults(vector<double>& resultVals) const
{

	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}

}
