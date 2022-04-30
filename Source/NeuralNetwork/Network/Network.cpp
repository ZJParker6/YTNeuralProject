#include "../../../Includes/NeuralNetwork/Network/Network.h"

Network::Network()
{
}

Network::~Network()
{
}

void Network::InitNeuralNetwork(uint32_t NumberOfInputsIn, uint32_t NumberOfOutputsIn, std::vector<uint32_t> NumberOfHiddenNeuronsIn, std::vector<EActivationMethod> HiddenMethodsIn, EActivationMethod OutputMethodIn)
{
	/* set class attributes */
	NumberOfInputs = NumberOfInputsIn;
	InputValues.clear();
	InputValues.resize(NumberOfInputs);

	NumberOfHiddenLayers = NumberOfHiddenNeuronsIn.size();
	HiddenLayersRef.resize(NumberOfHiddenLayers);
	NumberOfHiddenNeurons = NumberOfHiddenNeuronsIn;

	HiddenMethods.resize(HiddenMethodsIn.size());
	HiddenMethods = HiddenMethodsIn;

	NumberOfOutputs = NumberOfOutputsIn;
	OutputValues.clear();
	OutputValues.resize(NumberOfOutputs);
	OutputMethod = OutputMethodIn;

	/* Spawn the input layer */
	InputLayerRef = new InputLayer;
	InputLayerRef->BuildLayer(NumberOfInputs, NumberOfInputs, EActivationMethod::LINEAR); // fix
	printf("... Input Layer Constructed\n\n");

	/* Spawn hidden layers IF any */
	if (NumberOfHiddenLayers > 0)
	{
		for (size_t i = 0; i < NumberOfHiddenLayers; i++)
		{
			HiddenLayersRef[i] = new HiddenLayer;

			// check if this is the FIRST hidden layer.
			if (i == 0)
			{
				HiddenLayersRef.at(i)->BuildLayer(NumberOfHiddenNeurons[i], NumberOfInputs, HiddenMethods[i]);
				InputLayerRef->SetNextLayer(HiddenLayersRef[i]);
			}
			// create all other hidden layers
			else
			{
				HiddenLayersRef[i]->BuildLayer(NumberOfHiddenNeurons[i], HiddenLayersRef[i - 1]->GetNumberOfNeuronsInLayer(), HiddenMethods[i]);
				HiddenLayersRef[i - 1]->SetNextLayer(HiddenLayersRef[i]);
			}
			printf("\nHidden Layer %d has been built", i);
		}
		printf("\n... Hidden Layer[s] Constructed\n\n");
	}
	else
	{
		printf("... No Hidden Layers to Construct\n\n");
	}

 

	/* Spawn output layer */
	OutputLayerRef = new OutputLayer;
	if (NumberOfHiddenLayers > 0)
	{
		OutputLayerRef->BuildLayer(NumberOfOutputs, HiddenLayersRef.back()->GetNumberOfNeuronsInLayer(), OutputMethod);
		OutputLayerRef->SetPreviousLayer(HiddenLayersRef.back()); // HiddenLayersRef[NumberOfHiddenLayers - 1]
	}
	else
	{
		OutputLayerRef->BuildLayer(NumberOfOutputs, InputLayerRef->GetNumberOfNeuronsInLayer(), OutputMethod);
		OutputLayerRef->SetPreviousLayer(InputLayerRef); // maybe switch to InputLayerRef->SetNextLayer(OutputLayerRef);
	}
	printf("... Output Layer Constructed\n\n");

}
// InputLayer::BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn)

void Network::SetInputs(std::vector<double> InputsIn)
{
	for (size_t i = 0; i < NumberOfInputs; i++)
	{
		InputValues[i] = InputsIn[i];
	}
}

void Network::NetworkCalc()
{
	InputLayerRef->SetInputs(InputValues);
	InputLayerRef->Calc();

	if (NumberOfHiddenLayers > 0) 
	{
		for (size_t i = 0; i < NumberOfHiddenLayers; i++)
		{
			HiddenLayersRef[i]->SetInputs(HiddenLayersRef[i]->GetPreviousLayer().GetOutputs());
			HiddenLayersRef[i]->Calc();
		}
	}



	for (size_t i = 0; i < NumberOfOutputs; i++)
	{
		OutputLayerRef->SetInputs(OutputLayerRef->GetPreviousLayer().GetOutputs());
		OutputLayerRef->Calc();
		OutputValues[i] = OutputLayerRef->GetOutput(i);
	}
}
