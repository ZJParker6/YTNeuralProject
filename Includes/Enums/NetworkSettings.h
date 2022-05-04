/**************************************************************************
*  SETTINGS - ENUMS
*
* A collection of enums (maybe structs) that are used for Neural Network Settings.
*
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once

enum class EActivationMethod {STEP, LINEAR, SIGMOID, HYPERTAN };

enum class ERandomizerSettings{XOR, LCG, MT};

enum class ELearningMode {ONLINE, BATCH};

enum class ELearningParadigm {SUPERVISED, UNSUPERVISED};

enum class ELossMeasurement { SIMPLE, SSE, NDEGREES, MSE, MAE, HUBER, POWER };
