#pragma once

#include <string>
#include <stdlib.h>
#include <stdio.h> // fprint
#include <cstring>
#include <string.h>


namespace UStream
{
	struct Data
	{
		/* 2D Array (doubles) of input */
		double** in;
		/* 2D array (doubles) for expected values (targets) */
		double** tg;
		/* Number of inputs */
		uint32_t nInputs;
		/* Number of output neurons */
		uint32_t nOutput;
		/* Number of features (or rows) */
		uint32_t rows;
	};


	/************************************************
	* METHODS
	************************************************/
	bool CompareStrings(std::string& Str1, std::string& Str2);

	bool CompareStringsSensitive(std::string& Str1, std::string& Str2);

	/* return the number of lines in a file */
	uint32_t lns(FILE* const file);

	/* Read the line from the file */
	char* ReadLine(FILE* const file);

	/* create a 2D array for the network to examine/test/train with */
	double** New2D(const uint32_t rowsIn, const uint32_t colsIn);

	/* Create new Data type */
	Data nData(const uint32_t NumberOfInputsIn, const uint32_t NumberOfOutputsIn, const uint32_t RowsIn);

	/* Prasing data that was read in */
	void Parse(const Data dataIn, char* line, const uint32_t rowsIn);

	/* Free up memory */
	void dFree(const Data oldData);
}

