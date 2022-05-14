#include "../../Includes/Utilities/StreamUtils.h"

bool UStream::CompareStrings(std::string& Str1, std::string& Str2)
{
	return ((Str1.size() == Str2.size()) && std::equal(Str1.begin(), Str1.end(), Str2.begin(), [](char& C1, char& C2)
		{return (C1 == C2 || std::toupper(C1) == std::toupper(C2)); }));
}

bool UStream::CompareStringsSensitive(std::string& Str1, std::string& Str2)
{
	return ((Str1.size() == Str2.size()) && std::equal(Str1.begin(), Str1.end(), Str2.begin(), [](char& C1, char& C2)
		{return (C1 == C2); }));
}

uint32_t UStream::lns(FILE* const file)
{
	int ch{ EOF }; // EOF = End-of-File, it is 'defined' constant.
	uint32_t lines{ 0 };
	uint32_t pc{ '\n' };

	/* loop through to the end of the file */
	while ((ch = getc(file)) != EOF);
	{
		// Detect if we have reach the end o the line
		// if we have, we will increment the array
		if (ch == '\n')
		{
			lines++;
		}

		pc = ch;
	}
	if (pc != '\n')
	{
		lines++;
	}

	rewind(file);

	return lines;
}

char* UStream::ReadLine(FILE* const file)
{
	int ch{ EOF };
	uint32_t reads{ 0 };
	uint32_t size{ 128 }; // bits

	char* line = (char*)malloc((size) * sizeof(char)); 

	while ((ch = getc(file) != '\n' && ch != EOF))
	{
		line[reads++] = ch;

		if (reads + 1 == size)
		{
			// resize memory to match what is neeeded when allocation was too small (double size)
			line = (char*)realloc((line), (size *= 2) * sizeof(char));
		}

	}

	/* ends a null char at the end of a line*/
	line[reads] = '\0';

	return line;
}

double** UStream::New2D(const uint32_t rowsIn, const uint32_t colsIn)
{
	/* Allocate the rows */
	double** row = (double**)malloc((rowsIn) * sizeof(double));
	/* Allocate for columns in the row */
	for (size_t i = 0; i < rowsIn; i++)
	{
		row[i] = (double*)malloc((colsIn) * sizeof(double));
	}

	return row;
}

UStream::Data UStream::nData(const uint32_t NumberOfInputsIn, const uint32_t NumberOfOutputsIn, const uint32_t RowsIn)
{
	const Data data =
	{
		New2D(RowsIn, NumberOfInputsIn),
		New2D(RowsIn, NumberOfOutputsIn),
		NumberOfInputsIn,
		NumberOfOutputsIn,
		RowsIn
	};

	return data;
}

void UStream::Parse(const Data dataIn, char* line, const uint32_t rowsIn)
{
	const uint32_t cols = dataIn.nInputs + dataIn.nOutput;

	for (size_t i = 0; i < cols; i++)
	{
		/* Tokenize values */
		// deliminitor with space (csv = ',')
		char* NextToken{ NULL };
		const double val{ atof(strtok_s(i == 0 ? line : NULL, " ", &NextToken)) };

		/* store the values */
		if (i < dataIn.nInputs)
		{
			dataIn.in[rowsIn][i] = val;
		}
		else
		{
			dataIn.tg[rowsIn][i - dataIn.nInputs] = val;
		}
	}
}

void UStream::dFree(const Data oldData)
{
	for (size_t i = 0; i < oldData.rows; i++)
	{
		free(oldData.in[i]);
		free(oldData.tg[i]);
	}

	free(oldData.in);
	free(oldData.tg);
}

UStream::Data UStream::Build(const char* fPath, const uint32_t NumberOfInputsin, const uint32_t NumberOfOutputs)
{
	FILE* file;

	errno_t err = fopen_s(&file, fPath, "r");

	if (file == NULL)
	{
		printf("Could not open %s\n", fPath);
		printf("Dataset does not exist!\n");
		exit(1);
	}

	const uint32_t rows = lns(file);
	Data data = nData(NumberOfInputsin, NumberOfOutputs, rows);

	for (size_t i = 0; i < rows; i++)
	{
		char* line = ReadLine(file);
		Parse(data, line, i);
		free(line);
	}
	fclose(file);

	return data;
}

void UStream::RandomizeSet(const Data DataIn)
{
	for (size_t i = 0; i < DataIn.rows; i++)
	{
		const uint64_t k = UMath::SRandomVeryFast(0, false, DataIn.rows);

		// storing the original input and target data
		double* OT = DataIn.tg[i]; // store target output
		double* IT = DataIn.in[i]; // Store input

		DataIn.tg[i] = DataIn.tg[k]; // move target[k] to target[i]
		DataIn.tg[k] = OT; // move stored target into target[k]

		DataIn.in[i] = DataIn.in[k]; // move input[k] into input[i]
		DataIn.in[k] = IT; // move stored input into input[k]
	}
}
