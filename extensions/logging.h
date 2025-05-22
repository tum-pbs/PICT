
/* #define LOGGING

#ifdef LOGGING
#define PROFILING
#endif */

#ifdef LOGGING
#include <iostream>
#include <string>
#endif

#ifdef PROFILING
#include <chrono>
#endif


//--- Logging and Profiling ---

#define LOG_V3_XYZ(v) "(" << v.x << "," << v.y << "," << v.z << ")"
#define LOG_V4_XYZW(v) "(" << v.x << "," << v.y << "," << v.z  << "," << v.w << ")"
#define LOG_M44_COL(m) "[" << m[0][0] << "," << m[1][0] << "," << m[2][0] << "," << m[3][0] << ";\n" \
						   << m[0][1] << "," << m[1][1] << "," << m[2][1] << "," << m[3][1] << ";\n" \
						   << m[0][2] << "," << m[1][2] << "," << m[2][2] << "," << m[3][2] << ";\n" \
						   << m[0][3] << "," << m[1][3] << "," << m[2][3] << "," << m[3][3] << "]"

#ifdef LOG
#undef LOG
#endif
#ifdef LOGGING
#define LOG(msg) std::cout << __FILE__ << "[" << __LINE__ << "]: " << msg << std::endl
#else
#define LOG(msg)
#endif

#ifdef PROFILING
#include <chrono>
//no support for nesting for now.
auto start = std::chrono::high_resolution_clock::now();
__host__ void beginSample(){start = std::chrono::high_resolution_clock::now();}
__host__ void endSample(std::string name){
	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\'" << name << "\': " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-6) << "ms" << std::endl;
}
#define BEGIN_SAMPLE beginSample()
#define END_SAMPLE(name) endSample(name)
#else
#define BEGIN_SAMPLE
#define END_SAMPLE(name)
#endif

