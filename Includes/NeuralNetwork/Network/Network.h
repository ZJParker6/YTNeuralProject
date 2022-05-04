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
	/* Gets number of ouf outputs*/
	inline uint32_t GetNumberOfOutputNeurons() const { return NumberOfOutputs;  } 
	// inline uint32_t GetNumberOfOutputNeurons() const { return OutputLayerRef->GetNumberOfNeuronsInLayer(); }
	/* Gets the number of inputs into an output neuron at index i */
	inline uint32_t GetOutputNeuronsNumInputs(size_t i) const { return OutputLayerRef->GetNeuron(i).GetNumberOfInputs(); }

	/* Gets the number of hidden layers */
	inline uint32_t GetNumberOfHiddenLayers() const { return NumberOfHiddenLayers; }
	/* Gets the number of Neurons in a hidden layer (at index i) */
	inline uint32_t GetNumberOfHiddenNeurons(size_t i) const { return HiddenLayersRef.at(i)->GetNumberOfNeuronsInLayer();  } // update(?) to: NumberOfHiddenNeurons?
	/* Gets the number of inputs into a hidden layer */
	inline uint32_t GetHiddenNeuronsNumInputs(size_t i, size_t j) const { return HiddenLayersRef.at(i)->GetNeuron(j).GetNumberOfInputs(); }
	/* Returns the neuron at index i in the output layer*/
	inline Neuron GetOutputNeuron(const size_t i) const { return OutputLayerRef->GetNeuron(i); }

	/* Gets inputs */
	inline std::vector<double> GetInputs() const { return InputValues; }
	/* Gets input at index i */
	inline double GetInput(size_t i) const { return InputValues.at(i); }

};