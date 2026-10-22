#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int main(){size_t n=256<<20;char*a=malloc(n),*b=malloc(n);memset(a,1,n);memset(b,2,n);
struct timespec t0,t;clock_gettime(CLOCK_MONOTONIC,&t0);size_t by=0;
for(;;){memcpy(b,a,n);__asm__ volatile("" ::: "memory");a[by%n]^=b[(by+7)%n];by+=2*n;clock_gettime(CLOCK_MONOTONIC,&t);double dt=t.tv_sec-t0.tv_sec+(t.tv_nsec-t0.tv_nsec)/1e9;
if(dt>5){printf("%.2f GB/s (read+write)\n",by/dt/1e9);fflush(stdout);by=0;t0=t;}}}
