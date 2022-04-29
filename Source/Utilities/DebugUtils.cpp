#include "../../Includes/Utilities/DebugUtils.h"

void UDebug::WriteToDebugLog(std::string InputIn)
{
	FILE* file = fopen("/logs/debug.txt", "a");

	if (file)
	{
		//set data and time
		std::time_t Now = time(0); // current date/time based on current system

		char* DT = ctime(&Now);
		fprintf(file, "%s -- ", DT);
		//set message/log
		fprintf(file, "%s\n", InputIn);
	}
	else
	{
		printf("Failed to read or write Debug log");
		throw std::runtime_error("Debug Log error - could not read or write debug log. ");
	}

}
