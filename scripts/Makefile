all = data_gen

data_gen : data_gen.cc
	# g++ -fsanitize=address -fno-omit-frame-pointer -g $< -o $@
	g++ -O3 $< -o $@

clean:
	rm -rf *.bin
	rm -rf ${all}
