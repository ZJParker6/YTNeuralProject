/**************************************************************************
* NEURAL LAYER CLASS DECLARATION
*
*This class is the file for the neuron layer - a parent class for input, output, and hidden layers.
* The layers store the neurons, collate inputs and outputs. 
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include "../Neuron/NeuronMaster.h"
#include "../../Enums/NetworkSettings.h"

class NeuralLayer
{
	/************************************************
	* CONSTRUCTORS
	************************************************/
public:
	/* default constructor */
	NeuralLayer();

	/* destructor */
	virtual ~NeuralLayer();

	/************************************************
	* ATTRIBUTES
	************************************************/
private:
	/* Array of the neurons stored in this layer */
	std::vector<Neuron> Neurons;
	EActivationMethod eActivationMethod{ EActivationMethod::SIGMOID };

protected:
	/* How many neurons are in this layer */
	uint32_t NumberOfNeuronsInLayer{ 0 };
	/* The layer prior to this layer - the layer that 'feeds' into this layer */
	NeuralLayer* PreviousLayer{ nullptr };
	/* A referenc to self */
	NeuralLayer* SelfRef{ this };
	/* The layer after this layer - the layer that this layer 'feeds' into */
	NeuralLayer* NextLayer{ nullptr };
	/* The inputs vector from the previous layer */
	std::vector<double> Inputs;
	/* The outputs vector from this layer */
	std::vector<double> Outputs;
	/* The number of inputs that this layer can receive */
	uint32_t NumberOfInputs{ 0 };

/************************************************
* METHODS
************************************************/
	/************************************************
	* SET UP
	* - SETTERS
	************************************************/
public:
	/* Sets up properties for layer */
	virtual void BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn);
	/* Sets how many neurons are in the layer */
	void SetLayerDepth(uint32_t NumberOfNeuronsInLayerIn);
	/* Sets activation method for layer use */
	void SetActivationMethod(EActivationMethod MethodIn); // considering private or protected
	/* Initialies the layer*/
	virtual void InitLayer();
	/* Add an already created neuron at index i in this layer */
	virtual void SetNeuron(size_t i, Neuron NeuronIn);
private:
	/* Set a reference to self*/
	void SetSelf();

protected:
	/* Sets the previous layer */
	virtual void SetPreviousLayer(NeuralLayer* LayerIn) {};
	/* sets the next layer */
	virtual void SetNextLayer(NeuralLayer* LayerIn) {};

	/************************************************
	* SET UP
	* - GETTERS
	************************************************/
	/* Returns the number of neurons in this layer */
	inline uint32_t GetNumberOfNeuronsInLayer() const { return NumberOfNeuronsInLayer; }
	/* Returns the neuron vector List for this layer*/
	inline std::vector<Neuron> GetListOfNeurons() const { return Neurons; }
	/* return an individaul neuron from the vector at index i */
	inline Neuron GetNeuron(size_t i) const { return Neurons.at(i); }
	/* Returns previous layer */
	inline NeuralLayer GetPreviousLayer() const { if (PreviousLayer != nullptr) return *PreviousLayer; }
	/* Returns next layer */
	inline NeuralLayer GetNextLayer() const { if (NextLayer != nullptr) return *NextLayer; }
	/* Returns activation method */
	inline EActivationMethod GetActivationMethod() const { return eActivationMethod; }


	/************************************************
	* OUTPUT (incl. act, and before act).
	*- SETTERS
	************************************************/
	/* Calculate the layer's output */
	virtual void Calc();

	/************************************************
	* OUTPUT (incl. act, and before act).
	* - GETTERS
	************************************************/
	/* Return this layer's vector of outputs */
	inline std::vector<double> GetOutputs() const { return Outputs; }
	/* Returns an output at index i */
	inline double GetOutput(size_t i) const { return Outputs.at(i); }
};
