#include "../../../Includes/NeuralNetwork/Neuron/NeuronMaster.h"
#include "../../../Includes/Utilities/MathUtils.h"
#include "../../../Includes/Utilities/DebugUtils.h"


Neuron::Neuron()
{
}

Neuron::~Neuron()
{
}

void Neuron::BuildNeuron(uint32_t NumOfInputsIn, EActivationMethod MethodIn)
{
	SetNumberOfInputs(NumOfInputsIn);

	Bias = 1.0f; // set bias (change this later).

	SetWeightVector();
	SetActivationMethod(MethodIn);

}

void Neuron::SetNumberOfInputs(uint32_t NumOfInputsIn)
{
	NumberOfInputs = NumOfInputsIn;
	InputValues.clear();
	InputValues.resize(NumberOfInputs);
}

void Neuron::InitNeuron()
{
	/* Ensure there is data to use, or destroy this neuron*/
	if (NumberOfInputs > 0) // if 0, the neuron should fail out. 
	{
		InitRandWeights();
	}
	else
	{
		this->~Neuron();
	}
}

void Neuron::SetWeightVector()
{
	Weights.clear(); // remove garbage
	Weights.resize(NumberOfInputs + 1); // allocate space for weights in memory
}

void Neuron::InitRandWeights()
{
	for (size_t i = 0; i < NumberOfInputs; i++)
	{
		/* The next line is for debugging, if debugging uncomment the line*/
		Weights[i] = UMath::SRandomVeryFast(0, true);
		/* the next line is for practical use, if debugging comment this line */
		// Weights[i] =  = UMath::SRandomVeryFast();
	}
}

void Neuron::SetIndividualWeight(size_t i, double ValueIn)
{
	// Check if in valid range of index for weights
	if (i >= 0 && i < NumberOfInputs)
	{
		// check that the range actually exists 
		try
		{
			Weights[i] = ValueIn;
			throw 001;
		}
		catch (const std::out_of_range& ErrorString)
		{
			UDebug::WriteToDebugLog("Out of Range Error: " + *ErrorString.what());
			Weights.push_back(ValueIn); // attempt to resolve any out of scope issues
		}
	}
}

void Neuron::SetActivationMethod(EActivationMethod MethodIn)
{
	eActivationMethod = MethodIn;
}

void Neuron::SetActivationCoefficent(double ValueIn)
{
	a = ValueIn;
}

void Neuron::SetInputs(std::vector<double> ValuesIn)
{
	for (size_t i = 0; i < NumberOfInputs; i++)
	{
		InputValues[i] = ValuesIn[i];
	}
}

void Neuron::SetIndividualInput(size_t i, double ValueIn)
{
	// Check if in valid range of index for weights
	if (i >= 0 && i < NumberOfInputs)
	{
		// check that the range actually exists 
		try
		{
			InputValues[i] = ValueIn;
			throw 002;
		}
		catch (const std::out_of_range& ErrorString)
		{
			UDebug::WriteToDebugLog("Out of Range Error: " + *ErrorString.what());
			InputValues.push_back(ValueIn); // attempt to resolve any out of scope issues
		}
	}
}

void Neuron::SetOutput()
{
	// calcs output prior to activation (raw output) 

	OutputBeforeActivation = 0.0f; // clear junk from initial output value

	if (NumberOfInputs > 0) // check that this neuron instance has been initialized
	{
		if (!InputValues.empty() && !Weights.empty()) // check that the instance has valid variables
		{
			for (size_t i = 0; i <= NumberOfInputs; i++)
			{
				// sum { if i = last neuron -> bias is added in; else, input * weight }.
				OutputBeforeActivation += (i == NumberOfInputs ? Bias : InputValues.at(i) * Weights.at(i));
			}
		}
	}
	// calc actual output (post activation) 
	switch (eActivationMethod)
	{
	case EActivationMethod::STEP:
		Output = UMath::ActStep(OutputBeforeActivation);
		break;
	case EActivationMethod::LINEAR:
		Output = UMath::ActLinear(a, OutputBeforeActivation);
		break;
	case EActivationMethod::SIGMOID:
		Output = UMath::ActSigmoid(a, OutputBeforeActivation);
		break;
	case EActivationMethod::HYPERTAN:
		Output = UMath::ActHyperTan(a, OutputBeforeActivation);
		break;
	default:
		Output = UMath::ActSigmoid(a, OutputBeforeActivation);
		break;
	}
}

void Neuron::UpdateWeight(size_t i, double ValueIn)
{
	if (i >= 0 && i < NumberOfInputs)
		Weights[i] = ValueIn;
}
