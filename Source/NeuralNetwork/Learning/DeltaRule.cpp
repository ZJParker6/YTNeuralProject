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

DeltaRule::DeltaRule(Network* NetworkIn, UStream::Data DataIn)
{
	NeuralNet = NetworkIn;
	TrainingDataSet = DataIn;

	/*
		Error
		GeneralError
		OverallError
	*/
	for (size_t i = 0; i < TrainingDataSet.rows; i++)
	{
		GeneralError.resize(TrainingDataSet.rows);
		GeneralError.at(i) = 0;
		Error.resize(TrainingDataSet.rows);

		for (size_t j = 0; j < TrainingDataSet.nOutput; j++)
		{
			if (i == 0)
			{
				OverallError.at(j) = 0;
			}
			Error.at(i).resize(TrainingDataSet.nOutput);
			Error.at(i).at(j) = 0;
		}
	}
}

DeltaRule::DeltaRule(Network* NetworkIn, UStream::Data DataIn, ELearningMode LearningModeIn)
{
	NeuralNet = NetworkIn;
	TrainingDataSet = DataIn;
	LearningMode = LearningModeIn;
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

void DeltaRule::SetTrainingDataSet(UStream::Data DataIn)
{
	TrainingDataSet = DataIn;
	
	for (size_t i = 0; i < TrainingDataSet.rows ; i++)
	{
		TestingGeneralError.resize(TrainingDataSet.rows);
		TestingGeneralError.at(i) = 0;
		TestingError.resize(TrainingDataSet.rows);

		for (size_t j = 0; j < TrainingDataSet.nOutput ; j++)
		{
			if (i == 0)
			{
				TestingOverallError.at(j) = 0;
			}
			TestingError.at(i).resize(TrainingDataSet.nOutput);
			TestingError.at(i).at(j) = 0;
		}
	}
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
			size_t SizeLocal = NeuralNet->GetInputs().size();
			DerivativeResults.resize(SizeLocal);
			std::vector<std::vector<double>> InputLocal;
			std::vector<double> NthInput;
			NthInput.resize(TrainingDataSet.rows);
			InputLocal.resize(SizeLocal);

			// NeuronLocal.DerivativeBatch(TrainingDataSet.in);

			// getting useable version of the Training Dataset Inputs to pass to the neuron.
			for (size_t i = 0; i < SizeLocal; i++)
			{
				InputLocal[i].resize(SizeLocal);

				for (size_t k = 0; k < SizeLocal; k++)
				{
					double ValLocal = TrainingDataSet.in[i][k];
					InputLocal[i][k] = ValLocal;
				}
			}
			DerivativeResults = NeuronLocal.DerivativeBatch(InputLocal);

			if (InputIn < NeuronLocal.GetNumberOfInputs())
			{
				for (size_t i = 0; i < TrainingDataSet.rows; i++)
				{
					for (size_t k = 0; k < TrainingDataSet.nInputs; k++)
					{
						// this runs only once, which is how we can get away with this not resetting the values. 
						NthInput[k] = TrainingDataSet.in[i][k];
					}
				}
			}
			else
			{
				for (size_t i = 0; i < TrainingDataSet.rows; i++)
				{
					NthInput[i] = 0.0f;
				}
			}

			double MutliDerivResultNthInput{ 0.0f };
			for (size_t i = 0; i < TrainingDataSet.rows; i++)
			{
				MutliDerivResultNthInput += Error.at(i).at(NeuronIn) * NeuronLocal.Derivative(NeuralNet->GetInputs()) * NthInput[i]; 
			}

			DeltaWeight *= MutliDerivResultNthInput;
			break;
		}
		return NeuronLocal.GetWeight(InputIn) + DeltaWeight;

	}
}

double DeltaRule::CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn, double ErrorIn)
{
	if (LayerNumberIn > 0) // change to 1 if int not vector
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		double DeltaWeight = LearningRate*ErrorIn;
		Neuron NeuronLocal = NeuralNet->GetOutputNeuron(NeuronIn); // if not helper

		switch (LearningMode)
		{
		case ELearningMode::ONLINE:

			DeltaWeight *= NeuronLocal.Derivative(NeuralNet->GetInputs());

			if (InputIn < NeuronLocal.GetNumberOfInputs())
			{
				DeltaWeight *= NeuralNet->GetInput(InputIn);
			}

			break;

		case ELearningMode::BATCH:
			// come back later. 
			size_t SizeLocal = NeuralNet->GetInputs().size();
			DerivativeResults.resize(SizeLocal);
			std::vector<std::vector<double>> InputLocal;
			std::vector<double> NthInput;
			NthInput.resize(TrainingDataSet.rows);
			InputLocal.resize(SizeLocal);

			// NeuronLocal.DerivativeBatch(TrainingDataSet.in);

			// getting useable version of the Training Dataset Inputs to pass to the neuron.
			for (size_t i = 0; i < SizeLocal; i++)
			{
				InputLocal[i].resize(SizeLocal);

				for (size_t k = 0; k < SizeLocal; k++)
				{
					double ValLocal = TrainingDataSet.in[i][k];
					InputLocal[i][k] = ValLocal;
				}
			}
			DerivativeResults = NeuronLocal.DerivativeBatch(InputLocal);

			if (InputIn < NeuronLocal.GetNumberOfInputs())
			{
				for (size_t i = 0; i < TrainingDataSet.rows; i++)
				{
					for (size_t k = 0; k < TrainingDataSet.nInputs; k++)
					{
						// this runs only once, which is how we can get away with this not resetting the values. 
						NthInput[k] = TrainingDataSet.in[i][k];
					}
				}
			}
			else
			{
				for (size_t i = 0; i < TrainingDataSet.rows; i++)
				{
					NthInput[i] = 0.0f;
				}
			}

			double MutliDerivResultNthInput{ 0.0f };
			for (size_t i = 0; i < TrainingDataSet.rows; i++)
			{
				MutliDerivResultNthInput += Error.at(i).at(NeuronIn) * NeuronLocal.Derivative(NeuralNet->GetInputs()) * NthInput[i];
			}

			DeltaWeight *= MutliDerivResultNthInput;
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
			Forward(k);

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
				ApplyNewWeights();
				CurrentRecord = ++k;
				if (k >= TrainingDataSet.rows)
				{
					k = 0;
					CurrentRecord = 0;
					Epoch++;
				}

				Forward(k);

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
			Forward();

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
				ApplyNewWeights();
				Forward();

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

void DeltaRule::Forward()
{
	if (NeuralNet->GetNumberOfHiddenLayers() > 0)
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		std::vector<double> InputLocal;
		std::vector<std::vector<double>> ExpectedLocal;
		for (size_t i = 0; i < TrainingDataSet.rows; i++)
		{
			// Get from TrainingDataSet.in (2d vector) the "interior" vector at index i of the "outer" vector
			int SizeLocal = *(&TrainingDataSet.in[i] + 1) - TrainingDataSet.in[i];
			InputLocal.resize(SizeLocal);
			ExpectedLocal.resize(SizeLocal);

			ObservedOutput.resize(TrainingDataSet.rows);
			ObservedOutput[i].resize(TrainingDataSet.nOutput);
			ExpectedLocal.resize(TrainingDataSet.rows);
			ExpectedLocal[i].resize(TrainingDataSet.nOutput);


			for (size_t k = 0; k < SizeLocal; k++)
			{
				double ValLocal = TrainingDataSet.in[i][k];
				InputLocal.at(i) = ValLocal;
			}
			NeuralNet->SetInputs(InputLocal);
			NeuralNet->NetworkCalc();
		
			for (size_t k = 0; k < SizeLocal; k++)
			{
				double ValLocal = NeuralNet->GetOutput(k);
				ObservedOutput[i][k] = ValLocal;
				ExpectedLocal[i][k] = TrainingDataSet.tg[i][k];
			}
			GeneralError[i] = SetGeneralError(ExpectedLocal[i], ObservedOutput[i]);
	
			for(size_t j = 0; j < TrainingDataSet.nOutput; j++)
			{
				Error.at(i).at(j) = SetSimpleError(ExpectedLocal[i][j], ObservedOutput[i][j]);
			}
			// OverallGeneralError = overall general error algorithm (simple)
		}
		for (size_t j = 0; j < NeuralNet->GetNumberOfOutputNeurons(); j++)
		{
			OverallError[j] = SetOverallError(ExpectedLocal[j], InputLocal); // update later to Nth Values.
		}
		OverallGeneralError = SetOverallGeneralErrorList(ExpectedLocal, ObservedOutput);
	}
}

void DeltaRule::Forward(uint32_t i)
{
	if (NeuralNet->GetNumberOfHiddenLayers() > 0)
	{
		UDebug::WriteToDebugLog("Delta Rule cannot be applied with more than one layer in a neural network");
	}
	else
	{
		std::vector<double> InputLocal;
		std::vector<std::vector<double>> ExpectedLocal;
		int SizeLocal = *(&TrainingDataSet.in[i] + 1) - TrainingDataSet.in[i];
		InputLocal.resize(SizeLocal);
		ExpectedLocal.resize(SizeLocal);

		ObservedOutput.resize(TrainingDataSet.rows);
		ObservedOutput[i].resize(TrainingDataSet.nOutput);
		ExpectedLocal.resize(TrainingDataSet.rows);
		ExpectedLocal[i].resize(TrainingDataSet.nOutput);

		for (size_t k = 0; k < TrainingDataSet.nInputs; k++)
		{
			InputLocal[k] = TrainingDataSet.in[i][k];
		}
		NeuralNet->SetInputs(InputLocal);
		NeuralNet->NetworkCalc();

		for (size_t k = 0; k < SizeLocal; k++)
		{
			double ValLocal = NeuralNet->GetOutput(k);
			ObservedOutput[i][k] = ValLocal;
			ExpectedLocal[i][k] = TrainingDataSet.tg[i][k];
		}

		for (size_t k = 0; k < TrainingDataSet.nOutput; k++)
		{
			TrainingDataSet.tg[i][k] = NeuralNet->GetOutput(k);
		}

		GeneralError[i] = SetGeneralError(ExpectedLocal[i], ObservedOutput[i]);

		for (size_t j = 0; j < NeuralNet->GetNumberOfOutputNeurons(); j++)
		{
			OverallError[j] = SetOverallError(ExpectedLocal[j], InputLocal); // double check that this is not derivaties you are trying to use.
			Error[i][j] = SetSimpleError(ExpectedLocal[i][j], ObservedOutput[i][j]);
		}
		OverallGeneralError = SetOverallGeneralErrorList(ExpectedLocal, ObservedOutput);
	}
}

double DeltaRule::SetGeneralError(std::vector<double> YT, std::vector<double> YO)
{
	size_t Ny = YT.size();
	double ResultLocal{ 0.0f };

	// calculate individaul error variance
	for (size_t i = 0; i < Ny; i++)
	{
		ResultLocal += pow(YT[i] - YO[i], DegreeOverallError);
	}

	// return the standardized error (RSS, SSE)
	if (OverallErrorMeasurement == ELossMeasurement::MSE)
	{
		ResultLocal *= (1.0 / Ny);
	}
	// make sure to expand out with else if for the other "non-starter" algorithms
	else
	{
		ResultLocal *= (1.0 / DegreeOverallError);
	}

	return ResultLocal;
}

/*
* I combined the SetOverallGeneralError and the SetGeneralError
double DeltaRule::SetOverallGeneralError(std::vector<double> YT, std::vector<double> YO)
{
	size_t Ny = YT.size();
	double ResultLocal{ 0.0f };
	double YResultLocal{ 0.0f };

	for (size_t i = 0; i < Ny; i++)
	{
		YResultLocal += pow(YT[i] - YO[i], DegreeGeneralError);
	}

	if (GeneralErrorMeasurement == ELossMeasurement::MSE)
	{
		ResultLocal += pow((1.0f / Ny) * YResultLocal, DegreeOverallError);
	}
	else // make else if later forother measurement types
	{
		ResultLocal += pow((1.0f / DegreeGeneralError) * YResultLocal, DegreeOverallError);
	}

	return ((1.0f / Ny)* ResultLocal);
}
*/

double DeltaRule::SetOverallGeneralErrorList(std::vector<std::vector<double>> YT, std::vector<std::vector<double>> YO)
{
	size_t N = YT.size();
	size_t Ny = YT.at(0).size();
	double ResultLocal{ 0.0f };

	for (size_t i = 0; i < N; i++)
	{
		double YResultLocal{ 0.0f };
		for (size_t j = 0; j < Ny; j++)
		{
			YResultLocal = pow(YT.at(i).at(j) - YO.at(i).at(j), DegreeGeneralError);
		}
		if (GeneralErrorMeasurement == ELossMeasurement::MSE)
		{
			ResultLocal += pow((1.0 / Ny) * YResultLocal, DegreeOverallError);
		}
		else
		{
			ResultLocal += pow((1.0 / DegreeGeneralError) * YResultLocal, DegreeOverallError);
		}
	}
	if (OverallErrorMeasurement == ELossMeasurement::MSE)
	{
		ResultLocal *= (1.0 / N);
	}
	else
	{
		ResultLocal *= (1.0 / DegreeOverallError);
	}
	return ResultLocal;
}

double DeltaRule::SetOverallError(std::vector<double> YT, std::vector<double> YO)
{
	size_t N = YT.size();
	double ResultLocal{ 0.0f };

	for (size_t i = 0; i < N; i++)
	{
		ResultLocal += pow((YT.at(i) - YO.at(i)), DegreeOverallError);
	}

	if (GeneralErrorMeasurement == ELossMeasurement::MSE)
	{
		ResultLocal *= (1.0f/N);
	}
	else // make else if later forother measurement types
	{
		ResultLocal *= (1.0f / DegreeOverallError);
	}

	return ResultLocal;
}

double DeltaRule::SetSimpleError(double YT, double YO)
{
	return YT - YO;
}
