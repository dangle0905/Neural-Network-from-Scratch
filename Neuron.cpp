#include "Neuron.h"

double Neuron::eta = 0.15; //overall net learning rate, [0.0 ..1.0]
double Neuron::alpha = 0.5; //momentum, multiplier of last deltaWeight, [0.0..n]

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;
	//get the sum of the previous layers output (which are our inputs)
	//include the bias node from the previous layer.

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	//transferFunction is activationFunction
	m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	//sum of derivatives of weights
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}



void Neuron::updateInputWeights(Layer& prevLayer)
{
	//the weights to be updated are in the connection container
	//in the neurons in the preceding layer



	for (unsigned n = 0; n < prevLayer.size(); ++n) 
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

		/*
		n (eta) - overall net learning rate
			0.0 - slow learner
			0.2 - medium learner
			1.0 - reckless learner

		a (alpha) - momentum
			0.0 - no momentum
			0.5 - moderate momentum
		*/
	}
}

double Neuron::transferFunction(double x)
{
	//tanh - output rang [-1.0, 1.0]

	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	//tanh derivative
	return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	//sum our contributions of the erros at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;

	}

	return sum;

}
