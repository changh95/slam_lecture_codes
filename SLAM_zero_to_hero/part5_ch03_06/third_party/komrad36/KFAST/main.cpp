/*******************************************************************
*   main.cpp
*   KFAST
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Implementation of the FAST corner feature detector with optional
// non-maximal suppression, as described in the 2006 paper by
// Rosten and Drummond:
// "Machine learning for high-speed corner detection"
//         Edward Rosten and Tom Drummond
// https://www.edwardrosten.com/work/rosten_2006_machine.pdf
//
// My implementation uses AVX2, multithreading, and 
// many other careful optimizations to implement the
// FAST algorithm as described in the paper, but at great speed.
// This implementation outperforms the reference implementation by 40-60%
// single-threaded or 500% multi-threaded (!) while exactly matching
// the reference implementation's output and capabilities.
//
//
// Dr. Rosten's work is BSD-licensed:
//
// Copyright(c) 2006, 2008, 2009, 2010 Edward Rosten
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met :
// 
// 
// *Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// 
// *Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and / or other materials provided with the distribution.
// 
// *Neither the name of the University of Cambridge nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
// 	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// 	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// 	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
// 		NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// 	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "KFAST.h"

extern "C" {
#include "Rosten/fast.h"
}

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

void printReport(const std::string& name, const nanoseconds& dur, const size_t num_keypoints, const int width, const nanoseconds& comp = nanoseconds(0)) {
	std::cout << std::left << std::setprecision(6) << std::setw(10) << name << " took " << std::setw(7) << static_cast<double>(dur.count()) * 1e-3 << " us to find " << std::setw(width) << num_keypoints << (num_keypoints == 1 ? " keypoint" : " keypoints");
	if (comp.count() && comp < dur) {
		std::cout << " (" << std::setprecision(4) << std::setw(4) << static_cast<double>(dur.count()) / comp.count() << "x slower than KFAST)." << std::endl;
	}
	else {
		std::cout << '.' << std::endl;
	}
}

int main() {
	// ------------- Configuration ------------
	constexpr bool display_image = false;
	constexpr bool nonmax_suppress = true;
	constexpr auto warmups = 150;
	constexpr auto runs = 1000;
	constexpr auto thresh = 50;
	constexpr bool KFAST_multithread = true;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------


	// ------------- KFAST ------------
	std::vector<Keypoint> KFAST_kps;
	nanoseconds KFAST_ns;
	std::cout << std::endl << "------------- KFAST ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) KFAST<KFAST_multithread, nonmax_suppress>(image.data, image.cols, image.rows, static_cast<int>(image.step), KFAST_kps, thresh);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			KFAST<KFAST_multithread, nonmax_suppress>(image.data, image.cols, image.rows, static_cast<int>(image.step), KFAST_kps, thresh);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		KFAST_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- OpenCV ------------
	std::vector<cv::KeyPoint> CV_kps;
	nanoseconds CV_ns;
	std::cout << "------------ OpenCV ------------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) cv::FAST(image, CV_kps, thresh, nonmax_suppress);
	std::cout << "Testing..." << std::endl;
	{
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			cv::FAST(image, CV_kps, thresh, nonmax_suppress);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		CV_ns = (end - start) / runs;
	}
	// --------------------------------


	// ------------- Dr. Rosten ------------
	int R_size;
	xy* R_kps;
	nanoseconds R_ns;
	std::cout << "---------- Dr. Rosten ----------" << std::endl << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) {
		R_kps = (nonmax_suppress ? fast9_detect_nonmax : fast9_detect)(image.data, image.cols, image.rows, static_cast<int>(image.step), thresh, &R_size);
		free(R_kps);
	}
	{
		std::cout << "Testing..." << std::endl;
		const high_resolution_clock::time_point start = high_resolution_clock::now();
		for (int32_t i = 0; i < runs; ++i) {
			R_kps = (nonmax_suppress ? fast9_detect_nonmax : fast9_detect)(image.data, image.cols, image.rows, static_cast<int>(image.step), thresh, &R_size);
			if (i != runs - 1) free(R_kps);
		}
		const high_resolution_clock::time_point end = high_resolution_clock::now();
		R_ns = (end - start) / runs;
	}
	std::cout << "--------------------------------" << std::endl << std::endl << "Verifying results..." << std::endl << std::endl;
	// --------------------------------


	// ------------- Verification ------------
	if (CV_kps.size() != KFAST_kps.size() || CV_kps.size() != static_cast<size_t>(R_size)) {
		std::cerr << "ERROR! Sizes disagree!" << std::endl << std::endl;
	}
	else {
		int i = 0;
		for (; i < R_size; ++i) {
			if (CV_kps[i].pt.x != KFAST_kps[i].x || CV_kps[i].pt.y != KFAST_kps[i].y || CV_kps[i].pt.x != R_kps[i].x || CV_kps[i].pt.y != R_kps[i].y) {
				std::cerr << "ERROR! One or more points disagree!" << std::endl << std::endl;
				break;
			}
		}
		if (i == R_size) std::cout << "All keypoints agree! Test valid." << std::endl << std::endl;
	}
	free(R_kps);
	// --------------------------------

	
	// ------------- Output ------------
	const int max_width = static_cast<int>(ceil(log10(std::max(std::max(KFAST_kps.size(), CV_kps.size()), static_cast<size_t>(R_size)))));
	printReport("KFAST", KFAST_ns, KFAST_kps.size(), max_width);
	printReport("OpenCV", CV_ns, CV_kps.size(), max_width, KFAST_ns);
	printReport("Dr. Rosten", R_ns, R_size, max_width, KFAST_ns);
	std::cout << std::endl;
	if (display_image) {
		std::vector<cv::KeyPoint> converted_kps;
		for (const auto& kp : KFAST_kps) converted_kps.emplace_back(static_cast<float>(kp.x), static_cast<float>(kp.y), 0.0f, 0.0f, static_cast<float>(kp.score));
		cv::Mat image_with_kps;
		cv::drawKeypoints(image, converted_kps, image_with_kps, {255.0, 0.0, 0.0});
		cv::namedWindow("KFAST", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		cv::imshow("KFAST", image_with_kps);
		cv::waitKey(0);
	}
	// --------------------------------
}
