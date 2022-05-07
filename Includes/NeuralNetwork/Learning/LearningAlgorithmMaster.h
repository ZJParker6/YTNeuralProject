/**************************************************************************
* LEARNING ALGORITHM CLASS DECLARATION
*
* This class is the file the NN learning algorithms.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include <cstdint>
//#include "../Layers/NeuralLayerMaster.h"
#include "../Network/Network.h"
#include "../../Enums/NetworkSettings.h"
#include "../../Utilities/DebugUtils.h"

class LearningAlgorithm
{
	/************************************************
	* CONSTRUCTORS
	************************************************/
public:
	/* default constructor */
	LearningAlgorithm();
	/* Desuctor */
	~LearningAlgorithm();

	/************************************************
	* ATTRIBUTES
	************************************************/
protected:
	/* Network being trained */
	Network* NeuralNet;
	/* Online or Batch */
	ELearningMode LearningMode{ ELearningMode::ONLINE };
	/* Supervised or unsupervised */
	ELearningParadigm LearningParadigm{ ELearningParadigm::SUPERVISED };
	/* Maximum number of Epochs/Generations */
	uint32_t MaxEpochs{ 100 };
	/* Current Epoch/Generation */
	uint32_t Epoch{ 0 };
	/* Minimal Acceptablee Error */
	double MinOverallError{ 0.001 };
	/* Learning Rate */
	double LearningRate{ 0.1 };

	/* Should the training data be printed to screen */
	bool bPrintTraining{ false };

	/* Reference to the Hidden Layer */
	NeuralLayer* LayerRef{ nullptr };


	/************************************************
	* METHODS
	************************************************/
public:
	/************************************************
	* ABSTRACTED METHODS
	* DEFINED BY CHILD CLASSES
	************************************************/
	virtual void train() {};
	virtual void forward() {};
	virtual void forward(uint32_t i) {};
	virtual double CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn) { return 0.0f; }
	virtual double CalcNewWeight(uint32_t LayerNumberIn, uint32_t InputIn, const uint32_t NeuronIn, double ErrorIn) { return 0.0f; }
	virtual void Test() {};
	virtual void Test(uint32_t i) {};
	virtual void Print() {};

	/************************************************
	* GLOBAL TO ALL LEARNING SETUPS
	************************************************/
	/* Set max number of epochs/generations */
	void SetMaxEpochs(uint32_t x);
	/* Return max number of epochs/generations */
	inline uint32_t GetMaxEpochs() const { return MaxEpochs; }
	/* Returning current epoch/generation */
	inline uint32_t GetEpoch() const { return Epoch; }

	/* Set minimum overall error */
	void SetMinOverallError(double x);
	/* Get minimum overall error */
	inline double GetMinOverallError() const { return MinOverallError; }

	/* Set learning rate per epoch */
	void SetLearningRate(double x);
	/* Return learning rate*/
	inline double GetLearningRate() const { return LearningRate; }

	/* Sets learning mode */
	void SetLearningMode(ELearningMode x);
	/* Returns learning mode */
	inline ELearningMode GetLearningMode() const { return LearningMode; }

	/* Set learning paradigm */
	void SetLearningParadigm(ELearningParadigm x);
	/* return learning paradigm */
	inline ELearningParadigm GetLearningParadigm() const { return LearningParadigm; }
};
