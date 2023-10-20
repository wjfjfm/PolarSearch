#!/bin/bash

cd hnswlib
cmake .
make -j8

ls example_*
