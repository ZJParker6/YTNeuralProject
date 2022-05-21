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
#include "../../Utilities/StreamUtils.h"
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
	DeltaRule(Network* NetworkIn, UStream::Data DataIn);
	DeltaRule(Network* NetworkIn, UStream::Data DataIn, ELearningMode LearningModeIn);
	/* Desuctor */
	~DeltaRule();


/************************************************
* ATTRIBUTES
************************************************/
public:
	std::vector<std::vector<double>> ObservedOutput;

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
	UStream::Data TrainingDataSet;
	std::vector<double>  DerivativeResults;


/************************************************
* METHODS
************************************************/
public:
	/* Sets the General Error Measurement - and the relevent degrees */
	void SetGeneralErrorMeasure(ELossMeasurement ErrorType);
	/* Sets the Overall Error Measurement - and the relevent degrees */
	void SetOverallErrorMeasure(ELossMeasurement ErrorType);

	/* Parses the dataset for learning algorithm*/
	void SetTrainingDataSet(UStream::Data DataIn);

	/* Calculates the new weights during training for the network to update to - using learning rate. */
	double CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn) override;
	double CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn, double ErrorIn) override;

	/* Runs training paradigm */
	void Train() override;
	/* Applying the new weights, determined by Train() */
	void ApplyNewWeights();

	void Forward() override;
	void Forward(uint32_t i) override;

	double SetGeneralError(std::vector<double> YT, std::vector<double> YO);
//	double SetOverallGeneralError(std::vector<double> YT, std::vector<double> YO);
	double SetOverallGeneralErrorList(std::vector<std::vector<double>> YT, std::vector<std::vector<double>> YO);
	double SetOverallError(std::vector<double> YT, std::vector<double> YO);
	double SetSimpleError(double YT, double YO);

};