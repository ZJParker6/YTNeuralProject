#include "../../../Includes/NeuralNetwork/Layers/HiddenLayer.h"

HiddenLayer::HiddenLayer()
{
}

HiddenLayer::~HiddenLayer()
{
}

void HiddenLayer::BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn)
{
	NeuralLayer::BuildLayer(NumberOfNeuronsInLayerIn, NumberOfInputsIn, MethodIn);
	InitLayer();
}

void HiddenLayer::SetPreviousLayer(NeuralLayer* LayerIn)
{
	if (LayerIn != nullptr)
	{
		PreviousLayer = LayerIn;
	}
}

void HiddenLayer::SetNextLayer(NeuralLayer* LayerIn)
{
	if (LayerIn != nullptr)
	{
		NextLayer = LayerIn;
		NextLayer->SetPreviousLayer(this);
	}
}
