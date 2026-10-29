/*******************************************************************
*   main.cpp
*   RGB2Y
*
*	Author: Kareem Omar
*	kareem.h.omar@gmail.com
*	https://github.com/komrad36
*
*	Last updated Sep 29, 2018
*******************************************************************/
//
// Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
// Roughly 3x faster than OpenCV's implementation with AVX2, or 2x faster
// than OpenCV's implementation if using SSE only.
//
// Converts an RGB color image to grayscale.
//
// You can use equal weighting by calling the templated
// function with weight set to 'false', or you
// can specify non-equal weights (only slightly slower).
//
// The default non-equal weights match OpenCV's default.
//
// For even more speed see the CUDA version:
// github.com/komrad36/CUDARGB2Y
//
// If you do not have AVX2, uncomment the #define NO_AVX_PLEASE below to
// use only SSE isntructions. NOTE THAT THIS IS ABOUT 50% SLOWER.
// A processor with full AVX2 support is highly recommended.
//
// All functionality is contained in RGB2Y.h.
// 'main.cpp' is a demo and test harness.
//

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "RGB2Y.h"

using namespace std::chrono;

void printReport(const std::string& name, const nanoseconds& dur, const nanoseconds& comp = nanoseconds(0)) {
	std::cout << std::left << std::setprecision(6) << std::setw(10) << name << " took " << std::setw(7) << static_cast<double>(dur.count()) * 1e-3 << " us";
	if (comp.count() && comp < dur) {
		std::cout << " (" << std::setprecision(4) << std::setw(4) << static_cast<double>(dur.count()) / comp.count() << "x slower than RGB2Y)." << std::endl;
	}
	else {
		std::cout << '.' << std::endl;
	}
}

int main() {
	// ------------- Configuration ------------
	constexpr auto warmups = 50;
	constexpr auto runs = 100;

	// OpenCV's impl is multithreaded so set to true for fair comparison
	constexpr bool multithread = true;

	constexpr bool weighted_averaging = true;

	constexpr char name[] = "test.jpg";
	constexpr bool generate_test_pixels = false;
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image;
	if (!generate_test_pixels) {
		image = cv::imread(name);
		if (!image.data) {
			std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
			return EXIT_FAILURE;
		}
	}
	else {
		constexpr int patch_size = 3 * 256 * 256 * 256;
		image = cv::Mat(256, 256 * 256, CV_8UC3, new uint8_t[patch_size + 128]);
		int x = 0;
		for (int i = 0; i < 256; ++i) {
			for (int j = 0; j < 256; ++j) {
				for (int k = 0; k < 256; ++k) {
					image.data[x++] = i;
					image.data[x++] = j;
					image.data[x++] = k;
				}
			}
		}
	}
	// --------------------------------


	// ------------- Ref ------------
	nanoseconds RGB2Y_ref_ns;
	uint8_t* refimage = new uint8_t[image.cols*image.rows];
	std::cout << std::endl << "------------- Ref ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < 0; ++i) RGB2Y_ref<weighted_averaging>(image.data, image.cols, image.rows, image.cols, refimage);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < 1; ++i) {
			RGB2Y_ref<weighted_averaging>(image.data, image.cols, image.rows, image.cols, refimage);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		RGB2Y_ref_ns = end - start;
	}
	// --------------------------------


	// ------------- RGB2Y ------------
	nanoseconds RGB2Y_ns;
	uint8_t* rgb2y_image = new uint8_t[image.cols*image.rows];
	std::cout << "------------- RGB2Y ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) RGB2Y<multithread, weighted_averaging>(image.data, image.cols, image.rows, image.cols, rgb2y_image);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			RGB2Y<multithread, weighted_averaging>(image.data, image.cols, image.rows, image.cols, rgb2y_image);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		RGB2Y_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- OpenCV ------------
	nanoseconds CV_ns;
	cv::Mat newimage_cv;
	if (weighted_averaging) {
		std::cout << "------------ OpenCV ------------" << std::endl << "Warming up..." << std::endl;
		for (int i = 0; i < warmups; ++i) cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
		std::cout << "Testing..." << std::endl;
		{
			const high_resolution_clock::time_point start = high_resolution_clock::now();
			for (int32_t i = 0; i < runs; ++i) {
				cv::cvtColor(image, newimage_cv, CV_BGR2GRAY);
			}
			const high_resolution_clock::time_point end = high_resolution_clock::now();
			CV_ns = (end - start) / runs;
		}
	}
	// --------------------------------


	// ------------- Verification ------------
	int i = 0;
	size_t count = 0;
	std::cout << std::endl;
	for (; i < image.cols*image.rows; ++i) {
		if ((int)refimage[i] != (int)rgb2y_image[i]) {
			//std::cerr << "ERROR! One or more pixels disagree!" << std::endl;
			std::cerr << i << ": got " << +rgb2y_image[i] << ", should be " << /*(weighted_averaging ? */+refimage[i]/* : +refresult[i])*/ << std::endl;
			++count;
			//break;
		}
	}
	//if (i == image.cols*image.rows) std::cout << "All pixels agree! Test valid." << std::endl << std::endl;
	/*else */std::cout << count << " pixels disagree." << std::endl << std::endl;
	// --------------------------------


	// ------------- Output ------------
	printReport("Ref", RGB2Y_ref_ns);
	printReport("RGB2Y", RGB2Y_ns);
	if (weighted_averaging)	printReport("OpenCV", CV_ns, RGB2Y_ns);
	std::cout << std::endl;
	// --------------------------------
}
