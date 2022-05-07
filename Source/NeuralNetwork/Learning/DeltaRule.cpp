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

void DeltaRule::Train()
{
	if (NeuralNet->GetNumberOfHiddenLayers() > 0)
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		switch (LearningMode)
		{
		case ELearningMode::ONLINE:
		{
			Epoch = 0;
			int k = 0;
			CurrentRecord = 0;
			forward(k);

			// print for debugging and for double check network functions in training/learning.
			if (bPrintTraining)
			{
				// come back to this.
			}

			while (Epoch<MaxEpochs && OverallGeneralError > MinOverallError)
			{
				for (size_t j = 0; j < NeuralNet->GetNumberOfOutputNeurons(); j++)
				{
					for (size_t i = 0; i <= NeuralNet->GetNumberOfInputs(); i++)
					{
						NewWeights.at(0).at(j) = CalcNewWeight(0, i, j);
					}
				}
				// APPLY new weight
				CurrentRecord = ++k;

				// come back to later (stored data).

							// print for debugging and for double check network functions in training/learning.
				if (bPrintTraining)
				{
					// come back to this.
				}
			}
		}
			break;
		case ELearningMode::BATCH:
			Epoch = 0;
			forward();

			// print for debugging and for double check network functions in training/learning.
			if (bPrintTraining)
			{
				// come back to this.
			}

			while (Epoch<MaxEpochs && OverallGeneralError > MinOverallError)
			{
				Epoch++;
					for (size_t j = 0; j < NeuralNet->GetNumberOfOutputNeurons(); j++)
					{
						for (size_t i = 0; i <= NeuralNet->GetNumberOfInputs(); i++)
						{
							NewWeights.at(0).at(j) = CalcNewWeight(0, i, j);
						}
					}
					// APPLY new weight
					forward();


					// print for debugging and for double check network functions in training/learning.
					if (bPrintTraining)
					{
						// come back to this.
					}
			}
			break;
		}
	}
}

void DeltaRule::ApplyNewWeights() // double check indexing of vector.
{
	uint32_t NumberOfHiddenLayers = NeuralNet->GetNumberOfHiddenLayers();

	for (size_t k = 0; k <= NumberOfHiddenLayers; k++)
	{
		LayerRef = nullptr;
		uint32_t NumberOfNeuronsInLayer{ 0 }, NumberOfInputsIntoNeuron{ 0 };
		if (k < NumberOfHiddenLayers)
		{
			LayerRef = NeuralNet->GetHiddenLayer(1);
			NumberOfNeuronsInLayer = LayerRef->GetNumberOfNeuronsInLayer();
			for (size_t j = 0; j < NumberOfNeuronsInLayer; j++)
			{
				NumberOfInputsIntoNeuron = LayerRef->GetNeuron(j).GetNumberOfInputs();

				for (size_t i = 0; i <= NumberOfInputsIntoNeuron; i++)
				{
					double NewWeightLocal = NewWeights.at(j).at(i);
					LayerRef->GetNeuron(j).UpdateWeight(i, NewWeightLocal);
				}
			}
		}
		else
		{
			LayerRef = NeuralNet->GetOutputLayer();
			NumberOfNeuronsInLayer = LayerRef->GetNumberOfNeuronsInLayer();
			for (size_t j = 0; j < NumberOfNeuronsInLayer; j++)
			{
				NumberOfInputsIntoNeuron = LayerRef->GetNeuron(j).GetNumberOfInputs();

				for (size_t i = 0; i <= NumberOfInputsIntoNeuron; i++)
				{
					double NewWeightLocal = NewWeights.at(j).at(i);
					LayerRef->GetNeuron(j).UpdateWeight(i, NewWeightLocal);
				}
			}

		}
	}
}

void DeltaRule::forward()
{
	if (NeuralNet->GetNumberOfHiddenLayers() > 0)
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		// TODO: get dataset imp done first.
	}
}

void DeltaRule::forward(uint32_t i)
{
	if (NeuralNet->GetNumberOfHiddenLayers() > 0)
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		// TODO: get dataset imp done first.
	}
}

