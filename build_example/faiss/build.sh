#!/bin/bash

#g++ --std=c++17 -o demo demo_imi_flat.cpp -I. -L./faiss_lib -lfaiss -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath-link=./faiss_lib -Wl,-rpath=./faiss_lib
g++ --std=c++17 -o demo demo_imi_flat.cpp -I. -L./faiss_lib -lfaiss  -Wl,-rpath-link=./faiss_lib -Wl,-rpath=./faiss_lib


