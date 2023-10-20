#!/bin/bash

g++ precision_test.cpp -DANNOYLIB_MULTITHREADED_BUILD -o demo -std=c++14 -pthread
