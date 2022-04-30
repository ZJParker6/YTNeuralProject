/**************************************************************************
* NEURAL HIDDEN LAYER CLASS DECLARATION
*
*This class will handle all hidden transform events for the network.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include "NeuralLayerMaster.h"

class HiddenLayer : public NeuralLayer
{
	/************************************************
	* CONSTRUCTORS
	************************************************/
public:
	/* Default constructor */
	HiddenLayer();
	/* Default destructor */
	~HiddenLayer();

	/************************************************
	* METHODS
	************************************************/
public:
	void BuildLayer(uint32_t NumberOfNeuronsInLayerIn, uint32_t NumberOfInputsIn, EActivationMethod MethodIn) override;
	virtual void SetPreviousLayer(NeuralLayer* LayerIn) override;
	virtual void SetNextLayer(NeuralLayer* LayerIn) override;
};