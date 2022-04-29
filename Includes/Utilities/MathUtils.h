/**************************************************************************
*  MATH UTILITY FUNCTION LIBRARY
* 
* A collection of math helper functions and structures
* 
* original code by Z. J. Parker ("Two Neurons") for his YT channel.
**************************************************************************/

#pragma once
#include <cmath>
#include <random>
#include <set>
#include "DebugUtils.h"


namespace UMath
{
	/**************************************************************************
	*  FUNCTION LIBRARY: COMMON MATH FUNCTIONS
	*  These functions are intended for general use across all classes
	*  There are no speific use cases for these functions.
	***************************************************************************/

	/* Return a ratio of a value, between 0.0 and 1.0f (inclusive) */
	inline float GetRatio(float RawValueIn, float MinimumIn, float MaximumIn)
	{
		if (RawValueIn >= MaximumIn)
			return 1.0f; // raw value is erqual to or greatrer than max, so ratio is 1.
		else if (RawValueIn > MinimumIn)
			return (RawValueIn - MinimumIn) / (MaximumIn - MinimumIn); // standard ratio.
		else
			return 0.0f; // catch-all
	}

	/* Returns a 1 or -1 based on the relation of 'T' to 0; 0 is returned as a 1 */
	template<class T>
	inline static T ValueSign(const T x)
	{
		return (x >= (T)0) ? (T)+1 : (T)-1;
	}

	/* Return a true for even and a false for odd */
	template<class T>
	inline bool CheckIsEven(T x)
	{
		if (std::fmod(x, 2) == 0)
			return true;
		else
			return false;
	}

	/**************************************************************************
	*  FUNCTION LIBRARY: PSEUDO-RANDOM NUMBER GENERATORS (PRNGS)
	*  These structs/functions are designed to generate PRNs.
	*  There are different approaches that can be used for different use cases
	***************************************************************************/

	inline uint64_t SeedGenerator(uint32_t SeedIn = 0, bool bUseOverrideIn = false)
	{
		if (!bUseOverrideIn)
		{
			// get a dstribution of random numbers
			std::random_device RDLocal;
			std::mt19937 generator(RDLocal());
			std::uniform_int_distribution<uint32_t> PossibleSeedsLocal(0, 10000);

			// set the seed value
			return PossibleSeedsLocal(generator);
		}
		else
		{
			// returns a seed provided by the user
			return SeedIn;
		}
	}

	/* XOR shift random number generator */
	struct SRandomVeryFast
	{
	private:
		uint64_t ShuffleTable[4]{ 0 };
		// uint64_t T{ 0 };
		uint64_t Result{ 0 };
		uint64_t XorShuffle(uint32_t SeedIn);
		uint32_t Seed{ 0 };
		bool bUseOverride{ false };

		int64_t MAX_RANDOM = 0x7fff; // 32767

	public:
		SRandomVeryFast(uint32_t SeedIn = 0, bool bUseOverrideIn = false, uint64_t MaxIn = 0x7fff)
		{
			Seed = SeedIn;
			bUseOverride = bUseOverrideIn;
			MAX_RANDOM = MaxIn;
		}

		// Conversion to float, return random float between 0 and 1 (1 exclusive)
		operator double()
		{
			return double(double(XorShuffle(SeedGenerator(Seed, bUseOverride))) * (1.0f / 314159265358979.0f));
		}

		// Conversion to int64, returns random int64 between 0 and MAX_RANDOM (max inclusive) */
		operator uint64_t()
		{
			return XorShuffle(SeedGenerator(Seed, bUseOverride)) & MAX_RANDOM;
		}
	};

	inline uint64_t SRandomVeryFast::XorShuffle(uint32_t SeedIn)
	{
		// This is the next() function from the original xorshift generators (with minor variations)
		ShuffleTable[0] = Seed;
		ShuffleTable[1] = 362436069;
		ShuffleTable[2] = 521288629;

		ShuffleTable[0] ^= ShuffleTable[0] << 16;
		ShuffleTable[0] ^= ShuffleTable[0] >> 5;
		ShuffleTable[0] ^= ShuffleTable[0] << 1;

		//T = ShuffleTable[0];
		ShuffleTable[3] = ShuffleTable[0];
		ShuffleTable[0] = ShuffleTable[1];
		ShuffleTable[1] = ShuffleTable[2];
		ShuffleTable[2] = ShuffleTable[3] ^ ShuffleTable[0] ^ ShuffleTable[1];

		Result = ShuffleTable[2];

		return Result;
	}

	/* Fast random number generator structure using the linear congruintial generator (LCG)*/
	struct SRandomFast
	{
	private:
		uint32_t Seed{ 0 };
		bool bUseOverride{ false };

		int64_t MAX_RANDOM = 0x7fff; // 32767

	public:
		uint64_t GetFastRandom(uint32_t SeedIn);

		SRandomFast(uint32_t SeedIn = 0, bool bUseOverrideIn = false, uint64_t MaxIn = 0x7fff)
		{
			Seed = SeedIn;
			bUseOverride = bUseOverrideIn;
			MAX_RANDOM = MaxIn;
		}

		// Conversion to float, return random float between 0 and 1 (1 exclusive)
		operator double()
		{
			return double(double(GetFastRandom(SeedGenerator(Seed, bUseOverride))) * (1.0f / 314159265358979.0f));
		}

		// Conversion to int64, returns random int64 between 0 and MAX_RANDOM (max inclusive) */
		operator uint64_t()
		{
			return GetFastRandom(SeedGenerator(Seed, bUseOverride)) & MAX_RANDOM;
		}
	};

	inline uint64_t SRandomFast::GetFastRandom(uint32_t SeedIn)
	{
		Seed = 1664525L * SeedIn + 1013904223L; // values from the LCG algorithim (y = mX + b)

		uint64_t IntLocal = 0x3f800000 | (0x007fffff & Seed);

		return ((*&IntLocal) - (double)1.0f);
	} 

	/* return a random double (0 to 1; 1 exclusive). Meant for minor application in other math ulitilities */
	inline double RandomDouble(bool bUseVeryFastIn = true)
	{
		// rreturns 0 to 999, then divide to get a range of 0 to ~1
		if (bUseVeryFastIn)
			return double(SRandomVeryFast() % 1000) / 1000;
		else
		{
			return double(SRandomFast() % 1000) / 1000;
		}
	}

	/* returns amount of number(s) from 0 to upper limit*/
	inline std::set<int> GetRandomDistinctNumbers(int UpperLimitIn, int AmountIn)
	{
		// declare a set
		std::set<int> GeneratedNumbers;

		//iterate through the set
		while (GeneratedNumbers.size() < AmountIn)
		{
			GeneratedNumbers.insert(SRandomVeryFast() % UpperLimitIn);
		}

		return GeneratedNumbers;
	}

	/**************************************************************************
	*  FUNCTION LIBRARY: RANDOM BOOLS
	*  These functions generate random bools
	*  There are different approaches that can be used for different use cases
	***************************************************************************/

	/* get a random bool (t/f) with 50% chance. */
	inline bool GetRandomBool()
	{
		double RN = RandomDouble();
		if (RN < .5)
			return true;
		else
			return false;
	}

	/* Get a random Bool with user inputed chance */
	inline bool GetRandomBool(double ProbabilityIn)
	{
		// checks range
		if (ProbabilityIn >= 1)
		{
			UDebug::WriteToDebugLog("Error: MathsUtils - Type: GetRandomBool(double ProbabilityIn), probability out of range (Greater or equal to 1). Returned true.");
			return true;
		}
		else if (ProbabilityIn < 0)
		{
			UDebug::WriteToDebugLog("Error: MathsUtils - Type: GetRandomBool(double ProbabilityIn), probability out of range (less than 0). Returned false.");
			return false;
		}
		else
		{
			double RN = RandomDouble();
			if (RN < ProbabilityIn)
				return true;
			else
				return false;
		}
	}

	/* gets a random bool with user inputed ratio */
	inline bool GetRandomBool(float RawValueIn, float MinimumIn, float MaximumIn)
	{
		double Probability = GetRatio(RawValueIn, MinimumIn, MaximumIn);

		double RN = RandomDouble();
		if (RN < Probability)
			return true;
		else
			return false;
	}


	/**************************************************************************
	*  FUNCTION LIBRARY: ACTIVATION FUNCTIONS
	*  These functions generate the output for a neuron
	*  
	***************************************************************************/
	inline double ActStep(double x)
	{
		if (x < 0)
			return 0.0f;
		else
			return 1.0f;
	}

	inline double ActLinear(double a, double x)
	{
		return a * x;
	}

	inline double ActSigmoid(double a, double x)
	{
		return 1.0 / (1.0 - exp(-a * x));
	}

	inline double ActHyperTan(double a, double x)
	{
		return (1.0 - exp(-a * x)) / (1.0 + exp(-a * x));
	}
};