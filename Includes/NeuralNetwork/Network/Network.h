/**************************************************************************
* NETWORK CLASS DECLARATION
*
* This class will control the network.
* It will be used to init layers and get outputs.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include "../Layers/NeuralLayerMaster.h"
#include "../Layers/InputLayer.h"
#include "../Layers/HiddenLayer.h"
#include "../Layers/OutputLayer.h"
#include "../../Enums/NetworkSettings.h"
#include <vector>

class Network
{
/************************************************
* CONSTRUCTORS
************************************************/
public:
	/* Default constructor */
	Network();
	/* Default destructor */
	~Network();


/************************************************
* ATTRIBUTES
************************************************/
private:
	/* Neural Net's input layer */
	InputLayer* InputLayerRef;
	/* Neural Net's hidden layers */
	std::vector<HiddenLayer*> HiddenLayersRef;
	/* Neural Net's output layer */
	OutputLayer* OutputLayerRef;
	/* Number of Inputs */
	uint32_t NumberOfInputs{ 0 };
	/* Number of Hidden Layers */
	uint32_t NumberOfHiddenLayers{ 0 };
	/* Numer of Hidden Neurons PER Layer */
	std::vector<uint32_t> NumberOfHiddenNeurons;
	/* Number of Outputs */
	uint32_t NumberOfOutputs{ 0 };

	/* Hidden layer activation method */
	std::vector<EActivationMethod> HiddenMethods;
	/* Output layer activation method */
	EActivationMethod OutputMethod{ EActivationMethod::SIGMOID };
	/* Vector of values used for input */
	std::vector<double> InputValues;
	/* Vector of values returned for output */
	std::vector<double> OutputValues;

/************************************************
* METHODS
************************************************/
public:
	/* Active the network */
	void InitNeuralNetwork(uint32_t NumberOfInputsIn, uint32_t NumberOfOutputsIn, std::vector<uint32_t> NumberOfHiddenNeuronsIn, std::vector<EActivationMethod> HiddenMethodsIn, EActivationMethod OutputMethodIn);
	/* Sets the inputs for feeding in values to input layer */
	void SetInputs(std::vector<double> InputsIn);
	/* Calc the output for each layer and forward the value to the next layer OR the user (if output layer) */
	void NetworkCalc();

	/* Gets outputs */
	inline std::vector<double> GetOutputs() const { return OutputValues; }
	/* Gets output at index i */
	inline double GetOutput(size_t i) const { return OutputValues.at(i); }


};