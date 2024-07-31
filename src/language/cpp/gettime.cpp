#include <sys/time.h>
#include <cstddef>
#include <cstdio>
/* Get the number of milliseconds since SDL library initialization. */
int main() {
  // return NDL_GetTicks();
  // return 1;
  while(1){
  struct timeval current_time;
  gettimeofday(&current_time, NULL);
  printf("tv_sec: %d, tv_usec: %d\n", current_time.tv_sec, current_time.tv_usec);

  }
  return 0;
}