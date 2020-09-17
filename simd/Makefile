all:
	gcc -O3 -mavx  -fopenmp -o fastomp simd.c 
	gcc  -mavx   -o fast simd.c 
	gcc   -o non_simd non_simd.c 
	gcc  -g -S -o non_simd_s non_simd.c 
	gcc -Wa,-adhln -g non_simd.c > helloworld.s
