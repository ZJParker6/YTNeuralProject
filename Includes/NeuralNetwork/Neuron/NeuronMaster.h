/**************************************************************************
* NEURON CLASS DECLARATION
* 
* This class file is the Neuron Class, which will be put into a vector as part of a neuron layer.
* A neuron recieve an input (outside source or from another neuron)
* An transformation is made  - input*weight + bias
* Transformed value is passed through an activation function (determined by settings in the network)
* The neuron passsees the information out (output final decision/value to user or output value to next neuron).
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include <vector>
#include "../../Enums/NetworkSettings.h"

class Neuron
{
/************************************************
* CONSTRUCTORS
************************************************/
public:
	/* default constructor */
	Neuron();

	/* destructor */
	~Neuron();

/************************************************
* ATTRIBUTES
************************************************/
private:
	/* Input into a particular neuron. Vector will be unsized - resized when called */
	std::vector<double> InputValues;
	/* The number of Inputs, the length of the InputValues vector. If 0, the neuron instance will not init. Default is 0*/
	uint32_t NumberOfInputs{ 0 };
	/* Ouput from this instance of the neuron. This vlaue will be added asn an input to the above vector in the next layer*/
	double Output{ 0.0f };
	/* Output from this neuron prior to activation function (e.g., input*weight+bias) */
	double OutputBeforeActivation{ 0.0f };
	/* The activation method - default is Sigmoid */
	EActivationMethod eActivationMethod{ EActivationMethod::SIGMOID };

protected:
	/* Weights assocaited with this neuron */
	std::vector<double> Weights;
	/* bias */
	double Bias{ 0.0f };
	/* Transform coeffiecent */
	double a{ 1.0f };


/************************************************
* METHODS
************************************************/

public: 
	/************************************************
	* SET UP 
	* - SETTERS
	************************************************/
	/* Sets primary variables for the neuron */
	void BuildNeuron(uint32_t NumOfInputsIn, EActivationMethod MethodIn);
	/* Sets initial values for weights and for bias */
	void InitNeuron();
	/* Sets number of inputs to this instance of the neuron */
	void SetNumberOfInputs(uint32_t NumOfInputsIn);
	/* Sets weight vector up for further use */
	void SetWeightVector();
	/* Sets random initial weights */
	void InitRandWeights();
	/* Set an individual weight value */
	void SetIndividualWeight(size_t i, double ValueIn);
	/* Sets the activation method for this neuron */
	void SetActivationMethod(EActivationMethod MethodIn);
	/* Sets the activation coefficient*/
	void SetActivationCoefficent(double ValueIn);

	/************************************************
	* SET UP
	* - GETTERS
	************************************************/
	/* Returns the Number of Inputs to a neuron */
	inline uint32_t GetNumberOfInputs() const { return NumberOfInputs; }
	/* Returns the activation method of this neuron */
	inline EActivationMethod GetActivationMethod() const { return eActivationMethod; }

	/************************************************
	* OUTPUT (incl. act, and before act).
	*- SETTERS
	************************************************/
	/* Set the InputValues by taking trhe values being passed in */
	void SetInputs(std::vector<double> ValuesIn);
	/* Set an individual input value */
	void SetIndividualInput(size_t i, double ValueIn);
	/* Calculate the Output Values */
	void SetOutput();

	/************************************************
	* OUTPUT (incl. act, and before act).
	*- GETTERS
	************************************************/
	/* Returns raw output/output before activation */
	inline double GetOutputBeforeActivation() const { return OutputBeforeActivation; }
	/* Returns activated output */
	inline double GetOutput() const { return Output;  }
	/* Returns the Weights Vector */
	inline std::vector<double> GetWeightsVector() const { return Weights; }
	/* Returns weight at index i*/
	inline double GetWeight(uint32_t i) const { return Weights.at(i); }

	/************************************************
	* TRAINING/LEARNING - SETTERS
	************************************************/
	/* set an individual weight value, this is done by the system at run time */
	void UpdateWeight(size_t i, double ValueIn);
};
