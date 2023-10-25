all = data_gen knng calc_dis

all: ${all}

calc_dis : calc_dis.cc
	# g++ -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -O3 $< -o $@

data_gen : data_gen.cc
	# g++ -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -O3 $< -o $@

knng : knng.cc
	g++ -mavx -mavx512f -mavx512dq -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	# g++ -mavx -mavx512f -mavx512dq -O3 $< -o $@

clean:
	rm -rf *.bin
	rm -rf ${all}
