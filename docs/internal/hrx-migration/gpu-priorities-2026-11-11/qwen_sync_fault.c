#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

int hipDeviceSynchronize(void) {
  static _Atomic unsigned calls;
  int (*real_sync)(void) = (int (*)(void))dlsym(RTLD_NEXT, "hipDeviceSynchronize");
  if (!real_sync) abort();
  unsigned call = atomic_fetch_add(&calls, 1) + 1;
  int status = real_sync();
  const char* fail = getenv("GUFO_SYNC_FAIL_AT");
  int inject = fail && call == strtoul(fail, NULL, 10);
  fprintf(stderr, "[sync-fault] call=%u real=%d injected=%d\n", call, status, inject);
  return inject && status == 0 ? 1 : status;
}
