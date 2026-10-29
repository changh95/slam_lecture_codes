# KFAST

![picture alt](https://raw.githubusercontent.com/komrad36/KFAST/master/results.jpg "Speed comparison of KFAST vs. other implementations")

KFAST is the fastest CPU implementation of the FAST feature detector for computer vision, as described in the 2006 paper by Rosten and Drummond:
 
"Machine learning for high-speed corner detection" by Edward Rosten and Tom Drummond
https://www.edwardrosten.com/work/rosten_2006_machine.pdf

Non-maximal suppression is supported. My implementation uses AVX2, multithreading, and many other careful optimizations to implement the FAST algorithm as described in the paper, but at great speed. This implementation outperforms the next-fastest implementation by 60% single-threaded or 500% multi-threaded (!) while exactly matching its output and capabilities.

The test driver requires OpenCV (for comparison and image load). KFAST itself requires AVX2 (for speeeeeeeeed!) but does NOT require OpenCV; it has no dependencies whatsoever.

Works on Linux (g++) and Windows (MSVC) with C++11 or higher.

Modern compiler and processor *highly* recommended.

On Linux you can do:

    make
    ./KFAST

All functionality is contained within 'KFAST.h'.
'main.cpp' is a test driver demonstrating a performance comparison between Dr. Rosten's code, OpenCV's implementation, and my implementation. Dr. Rosten's implementation is contained in the folder 'Rosten'.

Simply plugging KFAST into ORB results in a several-fold speed improvement; however, there are a LOT of other things that need to be improved about ORB too, and I am working on it as time permits. I'll release them all at once when finished.

I would like to thank Dr. Rosten for his brilliant FAST and FASTER corner detectors. You can visit his website at www.edwardrosten.com or his GitHub at https://github.com/edrosten.

Dr. Rosten's work is BSD-licensed:

Copyright (c) 2006, 2008, 2009, 2010 Edward Rosten
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

*Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

*Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

*Neither the name of the University of Cambridge nor the names of 
 its contributors may be used to endorse or promote products derived 
 from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
