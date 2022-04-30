#include "../../../Includes/NeuralNetwork/Layers/OutputLayer.h"

OutputLayer::OutputLayer()
{
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn)
{
	NeuralLayer::BuildLayer(NumberOfNeuronsInLayerIn, NumberOfInputsIn, MethodIn);
	NextLayer = NULL;
	InitLayer();
}

void OutputLayer::SetPreviousLayer(NeuralLayer* LayerIn)
{
	if (LayerIn != nullptr)
	{
		PreviousLayer = LayerIn;
	}
}

void OutputLayer::SetNextLayer(NeuralLayer* LayerIn)
{
	LayerIn = NULL; // clear garbage
	NextLayer = NULL;
}
