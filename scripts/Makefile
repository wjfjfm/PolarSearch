all = data_gen knng calc_dis knng_2 knng_3

all: ${all}

calc_dis : calc_dis.cc
	# g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -mavx -mavx512f -mavx512dq -O3 $< -o $@

data_gen : data_gen.cc
	# g++ -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -O3 $< -o $@

knng : knng.cc
	# g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -mavx -mavx512f -mavx512dq -O3 $< -o $@

knng_2 : knng_2.cc
	# g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	# g++ -I${PAPI_DIR}/include -L${PAPI_DIR}/lib -mavx -mavx512f -mavx512dq -O3 $< -o $@ -lpapi
	#g++ -funroll-loops -mavx -mavx512f -mavx512dq -O3 $< -o $@
	# g++ -mavx -mavx512f -mavx512dq -O3 -S $< -o $@
	g++ -funroll-loops -mavx -mavx512f -mavx512dq -Ofast -finline-functions -ftree-loop-optimize -ftree-vectorize -mavx -mavx2 -mfma -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -fno-strict-aliasing -fomit-frame-pointer -pthread -fopenmp -march=native -flto $< -o $@

#  fp16_test : fp16_test.cc
#  	# g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
#  	# g++ -I${PAPI_DIR}/include -L${PAPI_DIR}/lib -mavx -mavx512f -mavx512dq -O3 $< -o $@ -lpapi
#  	g++ -funroll-loops -mavx -mavx512f -mavx512dq -mavx512fp16 -O3 $< -o $@
#  	# g++ -mavx -mavx512f -mavx512dq -O3 -S $< -o $@

knng_3 : knng_3.cc
	# g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -mavx -mavx512f -mavx512dq -O3 $< -o $@

clean:
	rm -rf ${all}
