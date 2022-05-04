#include "../../../Includes/NeuralNetwork/Learning/LearningAlgorithmMaster.h"

LearningAlgorithm::LearningAlgorithm()
{
}

LearningAlgorithm::~LearningAlgorithm()
{
}

void LearningAlgorithm::SetMaxEpochs(uint32_t x)
{
	MaxEpochs = x;
}

void LearningAlgorithm::SetMinOverallError(double x)
{
	MinOverallError = x;
}

void LearningAlgorithm::SetLearningRate(double x)
{
	LearningRate = x;
}

void LearningAlgorithm::SetLearningMode(ELearningMode x)
{
	LearningMode = x;
}

void LearningAlgorithm::SetLearningParadigm(ELearningParadigm x)
{
	LearningParadigm = x;
}
