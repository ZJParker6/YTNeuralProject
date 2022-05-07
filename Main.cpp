#include <iostream>
#include <vector>
#include <iostream>
#include "Includes/NeuralNetwork/Network/Network.h"
#include "Includes/Enums/NetworkSettings.h"
#include "Includes/Utilities/MathUtils.h"
#include "Includes/Utilities/StreamUtils.h"

using namespace std;

void RunRandTest()
{
	std::cout << "SUPER FAST UINT64: \n";
	uint64_t FirstTest = UMath::SRandomVeryFast();
	std::cout << "Default: " << FirstTest << std::endl;

	uint64_t SecondTest = UMath::SRandomVeryFast(0, true);
	std::cout << "Seed 0: " << SecondTest << std::endl;

	uint64_t ThirdTest = UMath::SRandomVeryFast(324, true);
	std::cout << "seed: 324: " << ThirdTest << std::endl;

	std::cout << "SUPER FAST LOOP DOUBLE: \n";
	for (int i = 0; i < 5; i++)
	{
		double LoopTest = UMath::SRandomVeryFast();
		std::cout << "Default Loop [" << i << "] : " << LoopTest << std::endl;
	}

	std::cout << "\n\n\n\n\nFast UINT64:";
	FirstTest = UMath::SRandomFast();
	std::cout << "Default: " << FirstTest << std::endl;
	SecondTest = UMath::SRandomFast(0, true);
	std::cout << "Seed 0: " << SecondTest << std::endl;
	ThirdTest = UMath::SRandomFast(324, true);
	std::cout << "seed: 324: " << ThirdTest << std::endl;

	std::cout << "FAST LOOP DOUBLE: \n";
	for (int i = 0; i < 5; i++)
	{
		double LoopTest = UMath::SRandomFast();
		std::cout << "Default Loop [" << i << "] : " << LoopTest << std::endl;
	}
}

void RunNetwork()
{
	uint32_t NumberOfInputs{ 0 };
	uint32_t NumberOfOutputs{ 0 };
	uint32_t NumberOfHiddenLayers{ 0 };
	vector<uint32_t> NumberOfHiddenNeurons;
	vector<EActivationMethod> HiddenMethods;
	EActivationMethod OutputMethod{ EActivationMethod::LINEAR };

	string MethodLocal;
	string AStep{ "STEP" };
	string ALinear{ "LINEAR" };
	string ASigmoid{ "SIGMOID" };
	string AHypertan{ "HYPERTAN" };

	cout << "\n\n";
	cout << "\t=================================\n";
	cout << "\t==     Neural Network Setup    ==\n";
	cout << "\t=================================\n";
	cout << "\n-------------------------------------------------- \n\n";
	cout << "\n\nStarting Neural Network Boot... : \n\n Setting Number of Inputs (and Hidden Layer Values)\n";
	cout << "\n-------------------------------------------------- \n\n";


	// 2 | 3 | 1 | 2, 2, 2


	/* have user set up network */

	//set initial values and methods
	cout << "\nPlease enter the number of INPUTS: ";
	cin >> NumberOfInputs;


	cout << "\nPlease enter number of hidden LAYERS: ";
	cin >> NumberOfHiddenLayers;
	if (NumberOfHiddenLayers != 0)
	{
		NumberOfHiddenNeurons.resize(NumberOfHiddenLayers);
		HiddenMethods.resize(NumberOfHiddenLayers);

		for (size_t i = 0; i < NumberOfHiddenLayers; i++)
		{
			
			cout << "\nPlease Enter the number of Neurons in Hidden Layer [" << i << "]: ";
			cin >> NumberOfHiddenNeurons.at(i);

			cout << "\nPlease select the activation method for layer [" << i << "]'s neuron[s]";
			cout << " \nEnter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
			cin >> MethodLocal;

			if (MethodLocal.empty())
			{
				cout << "\nNo method entered.";
				cout << "\nPlease enter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
				cin >> MethodLocal;
			}
			else if (UStream::CompareStrings(AStep, MethodLocal))
			{
				HiddenMethods[i] = EActivationMethod::STEP;
			}
			else if (UStream::CompareStrings(ALinear, MethodLocal))
			{
				HiddenMethods[i] = EActivationMethod::LINEAR;
			}
			else if (UStream::CompareStrings(ASigmoid, MethodLocal))
			{
				HiddenMethods[i] = EActivationMethod::SIGMOID;
			}
			else if (UStream::CompareStrings(AHypertan, MethodLocal))
			{
				HiddenMethods[i] = EActivationMethod::HYPERTAN;
			}
			else
			{
				cout << "No valid method entered";
				cout << "\nPlease enter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
				cin >> MethodLocal;
			}
		}
		MethodLocal.empty();
	}

	cout << "\nPlease enter the number of PREDICTIONS: ";
	cin >> NumberOfOutputs;

	cout << "\nPlease select the activation method for the output layer's neuron[s]";
	cout << " \nEnter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
	cin >> MethodLocal;

	if (MethodLocal.empty())
	{
		cout << "\nNo method entered.";
		cout << "\nPlease enter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
		cin >> MethodLocal;
	}
	else if (UStream::CompareStrings(AStep, MethodLocal))
	{
		OutputMethod = EActivationMethod::STEP;
	}
	else if (UStream::CompareStrings(ALinear, MethodLocal))
	{
		OutputMethod = EActivationMethod::LINEAR;
	}
	else if (UStream::CompareStrings(ASigmoid, MethodLocal))
	{
		OutputMethod = EActivationMethod::SIGMOID;
	}
	else if (UStream::CompareStrings(AHypertan, MethodLocal))
	{
		OutputMethod = EActivationMethod::HYPERTAN;
	}
	else
	{
		cout << "\nNo valid method entered";
		cout << "\nPlease enter one of the following: STEP, LINEAR, SIGMOID, HYPERTAN: ";
		cin >> MethodLocal;
	}

	// Set initials weights
	uint32_t EntryMethod; // NOT REGRESSION! 
	cout << "\nSelect Initial Weight method:\n";
	cout << "Enter 1 for Random Initial weights\nEnter 2 for Manual entry\nEnter 3 to read in\nEnter 4 for Debug \n : ";
	cin >> EntryMethod;

	switch (EntryMethod)
	{
	case 1:
		cout << "\nSelect randomizer:\nEnter 1 for XORshift\nEnter 2 for LCG\nEnter 3 for MT\n : ";
		break;
	case 2:
		break;
	case 3: 
		break;
	case 4:

		break;
	default:
		break;
	}

	/* MAKE NETWORK */
	cout << "\n-------------------------------------------------- \n";
	cout << "\n Creating Neural Network\n";
	cout << "\n-------------------------------------------------- \n\n";
	Network NeuralNetwork;
	cout << "Neural Network Built\n\n";
	NeuralNetwork.InitNeuralNetwork(NumberOfInputs, NumberOfOutputs, NumberOfHiddenNeurons, HiddenMethods, OutputMethod);
	cout << "Neural Network Initiated.\n";

	/*************************************************************************
	* TEST ONE
	*************************************************************************/
	// Set the input values for the network and get the output
	vector<double> NeuralInputs{ 1.5f, 0.5f };
	vector<double> NeuralOutputs;
	NeuralOutputs.resize(NumberOfOutputs);
	for (size_t i = 0; i < NumberOfInputs; i++)
	{
		cout << "\nFeeding the value [" << NeuralInputs[i] << "] to the neural network\n";
	}

	NeuralNetwork.SetInputs(NeuralInputs);
	NeuralNetwork.NetworkCalc();
	cout << "\n\n";
	cout << "\t=================================\n";
	cout << "\t==   Neural Network Outputs 1  ==\n";
	cout << "\t=================================\n";
	for (size_t i = 0; i < NumberOfOutputs; i++)
	{
		NeuralOutputs[i] = NeuralNetwork.GetOutput(i);
		cout << NeuralOutputs[i] << " ";
	}

	/*************************************************************************
	* TEST TWO
	*************************************************************************/
	NeuralInputs[0] = 1.0f;
	NeuralInputs[1] = 2.1f;
	for (size_t i = 0; i < NumberOfInputs; i++)
	{
		cout << "\nFeeding the value [" << NeuralInputs[i] << "] to the neural network\n";
	}

	NeuralNetwork.SetInputs(NeuralInputs);
	NeuralNetwork.NetworkCalc();
	cout << "\n\n";
	cout << "\t=================================\n";
	cout << "\t==   Neural Network Outputs 2  ==\n";
	cout << "\t=================================\n";
	for (size_t i = 0; i < NumberOfOutputs; i++)
	{
		NeuralOutputs[i] = NeuralNetwork.GetOutput(i);
		cout << NeuralOutputs[i] << " "; // return the output(s)
	}
}


int main()
{

	RunNetwork();
	return 0;
}