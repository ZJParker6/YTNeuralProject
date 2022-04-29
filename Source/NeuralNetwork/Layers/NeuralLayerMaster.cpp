#include "../../../Includes/NeuralNetwork/Layers/NeuralLayerMaster.h"
#include "../../../Includes/Utilities/DebugUtils.h"
#include <stdexcept>


NeuralLayer::NeuralLayer()
{
}

NeuralLayer::~NeuralLayer()
{
}

void NeuralLayer::BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn)
{
	SetLayerDepth(NumberOfNeuronsInLayerIn);
	NumberOfInputs = NumberOfInputsIn;
	SetActivationMethod(MethodIn);
	SetSelf(); // double check
}

void NeuralLayer::SetLayerDepth(uint32_t NumberOfNeuronsInLayerIn)
{
	NumberOfNeuronsInLayer = NumberOfNeuronsInLayerIn;
	Neurons.clear();
	Neurons.resize(NumberOfNeuronsInLayer);

	Outputs.clear();
	Outputs.resize(NumberOfNeuronsInLayer);
}

void NeuralLayer::SetActivationMethod(EActivationMethod MethodIn)
{
	eActivationMethod = MethodIn;
}

void NeuralLayer::InitLayer()
{
	if (NumberOfNeuronsInLayer > 0)
	{
		for (size_t i = 0; i < NumberOfNeuronsInLayer; i++)
		{
			Neuron NeuronLocal; // Declare/construction the neuron
			NeuronLocal.BuildNeuron(NumberOfInputs, eActivationMethod); // Build() the neuron
			NeuronLocal.InitNeuron(); // init() the neuron

			// store the neuron in the Neurons vector
			Neurons.at(i) = NeuronLocal;
		}
	}
	else
	{
		this->~NeuralLayer();
	}
}

void NeuralLayer::SetNeuron(size_t i, Neuron NeuronIn)
{
	try
	{
		Neurons[i] = NeuronIn;
		throw 003;
	}
	catch (const std::out_of_range& ErrorString)
	{
		UDebug::WriteToDebugLog("Out of Range Error: " + *ErrorString.what());
		Neurons.push_back(NeuronIn);
	}
}

void NeuralLayer::SetSelf()
{
	if (!SelfRef)
	{
		SelfRef = this;
	}
}

void NeuralLayer::Calc()
{
	if (!Inputs.empty() && !Neurons.empty())
	{
		for (size_t i = 0; i < NumberOfNeuronsInLayer; i++)
		{
			Neurons[i].SetInputs(Inputs);
			Neurons[i].SetOutput();

			Outputs[i] = Neurons.at(i).GetOutput();
		}
	}
}
