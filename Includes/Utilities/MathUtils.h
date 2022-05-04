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
#include <limits>
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
		uint64_t T{ 0 };
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

		// Conversion to float, return random float between 0 and MAX_RANDOM (max inclusive)
		operator double()
		{
			// return double(double(XorShuffle(SeedGenerator(Seed, bUseOverride))) % MAX_RANDOM); //* (1.0f / 314159265358979.0f));
			// return double(double(XorShuffle(SeedGenerator(Seed, bUseOverride)))); // no max 
			// return double(double(XorShuffle(SeedGenerator(Seed, bUseOverride))) * (1.0f / 314159265358979.0f)); // default version
			return (fmod(double(XorShuffle(SeedGenerator(Seed, bUseOverride))), MAX_RANDOM))/ MAX_RANDOM;
		}

		// Conversion to int64, returns random int64 between 0 and MAX_RANDOM (max inclusive) */
		operator uint64_t()
		{
			return XorShuffle(SeedGenerator(Seed, bUseOverride)) % MAX_RANDOM;
		}
	};

	inline uint64_t SRandomVeryFast::XorShuffle(uint32_t SeedIn)
	{
		// This is the next() function from the original xorshift generators (with minor variations)
		ShuffleTable[0] = SeedIn;
		ShuffleTable[1] = 362436069;
		ShuffleTable[2] = 521288629;

		ShuffleTable[0] ^= ShuffleTable[0] << 16;
		ShuffleTable[0] ^= ShuffleTable[0] >> 5;
		ShuffleTable[0] ^= ShuffleTable[0] << 1;

		T = ShuffleTable[0];
		//ShuffleTable[3] = ShuffleTable[0];
		ShuffleTable[0] = ShuffleTable[1];
		ShuffleTable[1] = ShuffleTable[2];
		ShuffleTable[2] = T ^ ShuffleTable[0] ^ ShuffleTable[1];

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

		// Conversion to float, return random float between 0 and MAX_RANDOM (Max Inclusive)
		operator double()
		{
			// return double(double(GetFastRandom(SeedGenerator(Seed, bUseOverride))) * (1.0f / 314159265358979.0f)); // original
			return (fmod(double(GetFastRandom(SeedGenerator(Seed, bUseOverride))), MAX_RANDOM)) / MAX_RANDOM;
		}

		// Conversion to int64, returns random int64 between 0 and MAX_RANDOM (max inclusive) */
		operator uint64_t()
		{
			return GetFastRandom(SeedGenerator(Seed, bUseOverride)) % MAX_RANDOM;
		}
	};

	inline uint64_t SRandomFast::GetFastRandom(uint32_t SeedIn)
	{
		Seed = 1664525L * SeedIn + 1013904223L; // values from the LCG algorithim (y = mX + b)

		uint64_t IntLocal = 0x3f800000 | (0x007fffff & Seed);

		return ((*&IntLocal) - (double)1.0f);
	} 

	/* Slow random number generator using Mersenne Twister */
	struct SRandomSlow
	{
	private:
		uint32_t Seed{ 0 };
		bool bUseOverride{ false };

		int64_t MAX_RANDOM = 0x7fff; // 32767

		static const int N = 312;
		static const int M = 156;

		// constact vector a
		static const uint64_t MATRIX_A = 0x9908b0df; // xor_mask from the random library

		//Most significant w-r bits
		static const uint64_t UPPER_MASK = 0x80000000; //  -2147483648

		//Least significant w-r bits
		static const uint64_t LOWER_MASK = 0x7fffffff; //  2147483647

		// Temperating parameters
		static const uint64_t TEMPERING_MASK_B = 0x9d2c5680; // the standard tempering b mask from MT19937
		static const uint64_t TEMPERING_MASK_C = 0xefc60000; // the standard tempering c mask from MT19937
		static const uint64_t TEMPERING_SHIFT_U = 11; // from the random library
		static const uint64_t TEMPERING_SHIFT_S = 7; // from the random library
		static const uint64_t TEMPERING_SHIFT_T = 15; // from the random library
		static const uint64_t TEMPERING_SHIFT_L = 18; // from the random library

		// the state vector
		uint64_t mt[N];
		int64_t mti = N + 1;

		uint64_t RandomSlow();

	public:
		SRandomSlow(uint32_t SeedIn = 5489, bool bUseOverrideIn = false, uint64_t MaxIn = 0x7fff)
		{
			Seed = SeedIn;
			bUseOverride = bUseOverrideIn;
			MAX_RANDOM = MaxIn;
		}

		// Conversion to float, returns random float between 0 and MAX_RANDOM (Max inclusive)
		operator double()
		{
			return fmod(RandomSlow(), MAX_RANDOM) / MAX_RANDOM;
			// 	return (fmod(double(XorShuffle(SeedGenerator(Seed, bUseOverride))), MAX_RANDOM))/ MAX_RANDOM;

		}

		// Conversion to int64, returns random int64 between 0 and MAX_RANDOM (MAX_RANDOM inclusive)
		operator uint64_t()
		{
			return RandomSlow() & MAX_RANDOM;
		}
	};

	inline uint64_t SRandomSlow::RandomSlow()
	{
		uint64_t Result;
		static uint64_t mag01[2] = { 0x0, MATRIX_A };

		if (mti >= N)
		{
			uint64_t i;

			if (mti == N + 1)
			{
				Seed = 5489ULL;
			}

			for (i = 0; i < N - M; i++)
			{
				Result = (mt[i] & UPPER_MASK | mt[i + 1] & LOWER_MASK);
				mt[i] = mt[i + M] ^ (Result >> 1) ^ mag01[Result & 0x1];
			}

			for (; i < N - 1; i++)
			{
				Result = (mt[i] & UPPER_MASK | mt[i + 1] & LOWER_MASK);
				mt[i] = mt[i + (M - N)] ^ (Result >> 1) ^ mag01[Result & 0x1];
			}

			Result = (mt[N - 1] & UPPER_MASK | mt[0] & LOWER_MASK);
			mt[N - 1] = mt[M - 1] ^ (Result >> 1) ^ mag01[Result & 0x1];

			mti = 0;
		}

		Result = mt[mti++];
		Result ^= (Result >> TEMPERING_SHIFT_U);
		Result ^= (Result << TEMPERING_SHIFT_S) & TEMPERING_MASK_B;
		Result ^= (Result << TEMPERING_SHIFT_T) & TEMPERING_MASK_C;
		Result ^= (Result >> TEMPERING_SHIFT_L);

		return Result;
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

	/**************************************************************************
	*  FUNCTION LIBRARY: LEARNING DERIVATIVE FUNCTIONS
	*  These functions generate the output for a neuron
	*
	***************************************************************************/

	inline double DerStep(double x) 
	{ 
		if (x == 0)
		{
			return std::numeric_limits<double>::max();
			// 1.7e308
		}
		else
		{
			return 0.0f;
		}
	}

	inline double DerLinear(double a)
	{
		return a;
	}

	inline double DerSigmoid(double a, double x)
	{
		return ActSigmoid(a, x)*(1-ActSigmoid(a, x));
	}
	
	inline double DerHypertan(double a, double x)
	{
		return (1.0)-pow(ActHyperTan(a, x), 2.0);
	}

};