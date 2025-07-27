#pragma once

// C Library
#include <cstdlib>
#include <csignal>


// Standard Library
#include <iostream>
#include <fstream>
#include <string>
#include <locale>
#include <codecvt>
#include <vector>
#include <deque>
#include <valarray>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <numeric>
#include <thread>
#include <future>
#include <functional>
#include <typeindex>
#include <regex>
#include <filesystem>
#include <source_location>
#include <memory>
#include <chrono>
#include <chrono>
#include <cmath>
#include <filesystem> 

#if __has_include(<format>)
#include <format>
#else
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/chrono.h>

namespace std {
	using ::fmt::format;
}
#endif


// Windows API Library
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCKAPI_
#include <Windows.h>
#endif
// DirectX Library
#ifdef _WIN32
#include <d3d11.h>
//#include <wrl/client.h>
//#include <dxgi1_4.h>
//#include <dxgi.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

// WaveOut Library
#ifdef _WIN32
#include <mmeapi.h>

#pragma comment(lib, "Winmm.lib")
#endif

// OpenCV
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world460d.lib")
#else
#pragma comment(lib, "opencv_world460.lib")
#endif


// ONNXRUNTIME


#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/provider_options.h>

#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "onnxruntime_providers_cuda.lib")
#pragma comment(lib, "onnxruntime_providers_shared.lib")