#include "../../../Includes/NeuralNetwork/Learning/DeltaRule.h"

DeltaRule::DeltaRule()
{
}

DeltaRule::DeltaRule(Network* NetworkIn)
{
	LearningParadigm = ELearningParadigm::SUPERVISED; // set to supervised learning, update later. TODO
	NeuralNet = NetworkIn;

	uint32_t NumberOfHiddenLayers = NeuralNet->GetNumberOfHiddenLayers();

	for (size_t i = 0; i < NumberOfHiddenLayers; i++)
	{
		NewWeights.resize(NumberOfHiddenLayers);
		
		uint32_t NumberOfNeuronsInLayer{ 0 }, NumberOfInputsInNeuron{ 0 };

		if (i < NumberOfHiddenLayers)
		{
			NumberOfNeuronsInLayer = NeuralNet->GetNumberOfHiddenNeurons(i); // gets number of neurons at current layer
			
			NewWeights.at(i).resize(NumberOfNeuronsInLayer);
			
			for (size_t j = 0; j < NumberOfNeuronsInLayer; j++)
			{
				NumberOfInputsInNeuron = NeuralNet->GetHiddenNeuronsNumInputs(i, j);

				for (size_t k = 0; k <= NumberOfInputsInNeuron; k++)
				{
					NewWeights.at(i).at(j) = 0.0f;
				}
			}
		}
		else
		{
			NumberOfNeuronsInLayer = NeuralNet->GetNumberOfOutputNeurons();

			NewWeights.at(i).resize(NumberOfNeuronsInLayer);

			for (size_t j = 0; j < NumberOfNeuronsInLayer; j++)
			{

				NumberOfInputsInNeuron = NeuralNet->GetOutputNeuronsNumInputs(j);

				for (size_t k = 0; k <= NumberOfInputsInNeuron; k++)
				{
					NewWeights.at(i).at(j) = 0.0f;
				}
			}
		}
	}
}

DeltaRule::~DeltaRule()
{
}

void DeltaRule::SetGeneralErrorMeasure(ELossMeasurement ErrorType)
{
	switch (ErrorType)
	{
	case ELossMeasurement::SIMPLE:
		DegreeGeneralError = 1;
		break;
	case ELossMeasurement::SSE:
		DegreeGeneralError = 1;
		break;
	case ELossMeasurement::NDEGREES:
		break;
	case ELossMeasurement::MSE:
		DegreeGeneralError = 2;
		break;
	case ELossMeasurement::MAE:
		DegreeGeneralError = 1;
		break;
	case ELossMeasurement::HUBER:
		break;
	case ELossMeasurement::POWER:
		break;
	default:
		DegreeGeneralError = 2;
		break;
	}
	GeneralErrorMeasurement = ErrorType;
}

void DeltaRule::SetOverallErrorMeasure(ELossMeasurement ErrorType)
{
	switch (ErrorType)
	{
	case ELossMeasurement::SIMPLE:
		DegreeOverallError = 1;
		break;
	case ELossMeasurement::SSE:
		DegreeGeneralError = 1;
		break;
	case ELossMeasurement::NDEGREES:
		break;
	case ELossMeasurement::MSE:
		DegreeGeneralError = 2;
		break;
	case ELossMeasurement::MAE:
		DegreeGeneralError = 1;
		break;
	case ELossMeasurement::HUBER:
		break;
	case ELossMeasurement::POWER:
		break;
	default:
		DegreeGeneralError = 2;
		break;
	}
	OverallErrorMeasurement = ErrorType;
}

double DeltaRule::CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn)
{
	if (LayerNumberIn > 0) // change to 1 if int not vector
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		double DeltaWeight = LearningRate;
		Neuron NeuronLocal = NeuralNet->GetOutputNeuron(NeuronIn); // if not helper

		switch (LearningMode)
		{
		case ELearningMode::ONLINE:

			DeltaWeight *= Error.at(CurrentRecord).at(NeuronIn);
			DeltaWeight *= NeuronLocal.Derivative(NeuralNet->GetInputs());

			if (InputIn < NeuronLocal.GetNumberOfInputs())
			{
				DeltaWeight *= NeuralNet->GetInput(InputIn);
			}
			break;
		case ELearningMode::BATCH:
			// come back later. 
			break;
		}
		return NeuronLocal.GetWeight(InputIn) + DeltaWeight;
	}
}
