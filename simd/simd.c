#include <immintrin.h>
#include <time.h>
#define VEC_LOAD(wide , small_addr) wide = _mm256_load_ps(small_addr)
#define VEC_STORE(small_addr, wide) _mm256_store_ps(small_addr, wide)
#define REG_256  __m256

int main()
{
    int n = 100000;
    float * a = (float*) malloc(sizeof(float)*n);
    float * b = (float*) malloc(sizeof(float)*n);
    float * c1 = (float*) malloc(sizeof(float)*n);
    float * c2 = (float*) malloc(sizeof(float)*n);

    // warm up

    for (int i =0; i<16; i++)
    {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    for (int i=0; i<20; i++){
    for (int i=0 ;i <n; i++)
    {
        c2[i] = a[i] + b[i]/a[i];
    }}


    struct timeval start2, end2;

    gettimeofday(&start2, NULL);
    #pragma omp parallel for
    for(int k=0; k<10;k++){
    for (int i = 0; i < n; i += 8) 
    {
        REG_256 va,vb,vc;
        VEC_LOAD(va, a+i);       
        VEC_LOAD(vb, b+i);       
        vc =  _mm256_add_ps(vb, va); 
        _mm256_store_ps(c1+i,vc);
    }}
    gettimeofday(&end2, NULL);
 

    printf("%ld\n", ((end2.tv_sec * 1000000 + end2.tv_usec)
		  - (start2.tv_sec * 1000000 + start2.tv_usec)));
}

