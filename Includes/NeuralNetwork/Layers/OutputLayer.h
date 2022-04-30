/**************************************************************************
* NEURAL OUTPUT LAYER CLASS DECLARATION
*
*This class will handle all output related events for the network.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include "NeuralLayerMaster.h"

class OutputLayer : public NeuralLayer
{
/************************************************
* CONSTRUCTORS
************************************************/
public:
	/* Default constructor */
	OutputLayer();
	/* Default destructor */
	~OutputLayer();

/************************************************
* METHODS
************************************************/
public:
	void BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn) override;
	virtual void SetPreviousLayer(NeuralLayer* LayerIn) override;
	virtual void SetNextLayer(NeuralLayer* LayerIn) override;
};