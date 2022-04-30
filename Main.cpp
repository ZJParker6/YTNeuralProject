#include <iostream>
#include <vector>
#include <iostream>
#include "Includes/NeuralNetwork/Network/Network.h"
#include "Includes/Enums/NetworkSettings.h"
#include "Includes/Utilities/MathUtils.h"

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

int main()
{
	cout << "\n\n";
	cout << "\t=================================\n";
	cout << "\t==     Neural Network Setup    ==\n";
	cout << "\t=================================\n";
	cout << "\n-------------------------------------------------- \n\n";
	cout << "\n\nStarting Neural Network Boot... : \n\n Setting Number of Inputs (and Hidden Layer Values)\n";
	cout << "\n-------------------------------------------------- \n\n";

	uint32_t NumberOfInputs = 2;
	uint32_t NumberOfOutputs = 1;
	vector<uint32_t> NumberOfHiddenNeurons{ 2, 2, 2 };

	cout << "Using: " << NumberOfInputs << " input[s]\n\n";
	cout << "Using: " << NumberOfOutputs << " Output[s]\n\n";

	if (NumberOfHiddenNeurons.size() > 0)
	{
		cout << "Using: " << NumberOfHiddenNeurons.size() << " Hidden Layer[s]\n\n";
		for (size_t i = 0; i < NumberOfHiddenNeurons.size(); i++)
		{
			cout << "... Hidden Layer[" << i << "] houses: " << NumberOfHiddenNeurons[i] << " Neuron[s]\n\n";
		}
	}
	else
	{
		cout << "Using No Hidden Layers\n\n";
	}

	// set methods for layers
	vector<EActivationMethod> HiddenMethods;

	if (NumberOfHiddenNeurons.size() > 1)
	{
		HiddenMethods.resize(NumberOfHiddenNeurons.size());

		for (size_t i = 0; i < NumberOfHiddenNeurons.size(); i++)
		{
			HiddenMethods[i] = EActivationMethod::SIGMOID;
			cout << "Hidden Layer[" << i << "] is using: SIGMOID\n\n";
		}
	}

	//Set Output
	EActivationMethod OutputMethod{ EActivationMethod::LINEAR };
	cout << "Output Layer is using: LINEAR\n\n";

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
	for (size_t i =0; i < NumberOfOutputs; i++)
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
		cout << NeuralOutputs[i] << " ";
	}
	// return the output(s)
	return 0;
}