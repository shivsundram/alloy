#include <immintrin.h>
#include <time.h>
#define VEC_LOAD(wide , small_addr) wide = _mm256_load_ps(small_addr)
#define VEC_STORE(small_addr, wide) _mm256_store_ps(small_addr, wide)
#define REG_256  __m256

void loop(float*a, float*b, float*c,int n)
{
    
    for (int i=0 ;i <n; i++)
    {
        c[i] = a[i] + b[i];
    }


}


int main()
{
    int n = 100000;
    float * a = (float*) malloc(sizeof(float)*n);
    float * b = (float*) malloc(sizeof(float)*n);
    float * c1 = (float*) malloc(sizeof(float)*n);
    //float * c2 = (float*) malloc(sizeof(float)*n);
    struct timeval start1, end1;

    gettimeofday(&start1, NULL);
    for (int k =0; k<10;k++) loop(a,b,c1,n);
    gettimeofday(&end1, NULL);


    printf("%ld\n", ((end1.tv_sec * 1000000 + end1.tv_usec)
		  - (start1.tv_sec * 1000000 + start1.tv_usec)));


}
