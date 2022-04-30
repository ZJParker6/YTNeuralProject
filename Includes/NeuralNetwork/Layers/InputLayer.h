/**************************************************************************
* NEURAL INPUT LAYER CLASS DECLARATION
*
*This class will handle all input related events for the network.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include "NeuralLayerMaster.h"

class InputLayer : public NeuralLayer
{
/************************************************
* CONSTRUCTORS
************************************************/
public:
	/* default constructor */
	InputLayer();
	/* default destructor */
	~InputLayer();

/************************************************
* METHODS
************************************************/
public:
	void BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn) override;
	virtual void SetPreviousLayer(NeuralLayer* LayerIn) override;
	virtual void SetNextLayer(NeuralLayer* LayerIn) override;

};
