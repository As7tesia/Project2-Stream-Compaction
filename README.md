CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)


### Build notes

added `/Zc:preprocessor` to CMakeLists for MSVC builds
(and `-Xcompiler=/Zc:preprocessor` for the CUDA side). CUDA 13.3's Thrust/CCCL headers refuse to compile under
MSVC's traditional preprocessor and fail with a fatal `C1189` in `thrust.cu` without it.
