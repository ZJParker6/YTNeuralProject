#include "../../../Includes/NeuralNetwork/Layers/InputLayer.h"

InputLayer::InputLayer()
{
}

InputLayer::~InputLayer()
{
}

void InputLayer::BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn)
{
	NeuralLayer::BuildLayer(NumberOfNeuronsInLayerIn, NumberOfInputsIn, MethodIn);
	PreviousLayer = NULL; 
	InitLayer();
}

void InputLayer::SetPreviousLayer(NeuralLayer* LayerIn)
{
	LayerIn = NULL; //clearing junk
	PreviousLayer = NULL;
}

void InputLayer::SetNextLayer(NeuralLayer* LayerIn)
{
	if (LayerIn != nullptr)
	{
		NextLayer = LayerIn;
		NextLayer->SetPreviousLayer(this);
	}
}
