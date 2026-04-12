#include <stdint.h>

typedef struct {
    void* nextInChain;
    uint32_t mode;
    void* callback;
    void* userdata;
} WGPURequestAdapterCallbackInfo;

void call_rust(void* inst, void* opt, WGPURequestAdapterCallbackInfo info);

void test_call() {
    WGPURequestAdapterCallbackInfo info = {0, 2, (void*)0x1234, (void*)0x5678};
    call_rust((void*)1, (void*)2, info);
}
