/**************************************************************************
* LEARNING ALGORITHM CLASS DECLARATION
*
* This class controls how we update our weights for inputs -
* Using the delta rule (of course).
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include <vector>
#include "LearningAlgorithmMaster.h"
#include "../../Enums/NetworkSettings.h"
#include "../Neuron/NeuronMaster.h"

class DeltaRule : LearningAlgorithm
{
	/************************************************
	* CONSTRUCTORS
	************************************************/
public:
	/* default constructor */
	DeltaRule();
	DeltaRule(Network* NetworkIn);
	/* Desuctor */
	~DeltaRule();


/************************************************
* ATTRIBUTES
************************************************/
public:
	std::vector<std::vector<double>> Error;
	std::vector<double> GeneralError;
	std::vector<double> OverallError;
	double OverallGeneralError{ 0.0f };

	std::vector<std::vector<double>> TestingError;
	std::vector<double> TestingGeneralError;
	std::vector<double> TestingOverallError;
	double TestingOverallGeneralError{ 0.0f };

	double DegreeGeneralError{ 2.0f };
	double DegreeOverallError{ 0.0f };

	ELossMeasurement GeneralErrorMeasurement{ ELossMeasurement::SSE };
	ELossMeasurement OverallErrorMeasurement{ ELossMeasurement::MSE };

private:
	uint32_t CurrentRecord{ 0 };
	std::vector<std::vector<double>> NewWeights;



/************************************************
* METHODS
************************************************/
public:
	/* Sets the General Error Measurement - and the relevent degrees */
	void SetGeneralErrorMeasure(ELossMeasurement ErrorType);
	/* Sets the Overall Error Measurement - and the relevent degrees */
	void SetOverallErrorMeasure(ELossMeasurement ErrorType);

	double CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn) override;
};