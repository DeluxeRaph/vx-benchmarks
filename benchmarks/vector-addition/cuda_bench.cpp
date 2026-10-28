#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define RT(x) do { auto rc=(x); if(rc!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(rc)); exit(1); } } while(0)
#define DR(x) do { auto rc=(x); if(rc!=CUDA_SUCCESS) { const char* s; cuGetErrorName(rc,&s); fprintf(stderr,"Driver: %s\n",s); exit(1); } } while(0)
#define RTC(x) do { auto rc=(x); if(rc!=NVRTC_SUCCESS) { fprintf(stderr,"NVRTC: %s\n",nvrtcGetErrorString(rc)); exit(1); } } while(0)
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
int main(int argc, char** argv) {
    if(argc!=4) return 2;
    int rows=atoi(argv[1]), threads=atoi(argv[2]), timed=atoi(argv[3]);
    size_t n=size_t(rows)*threads, bytes=n*sizeof(float);
    RT(cudaFree(nullptr));
    std::ifstream file("cuda_add.cu");
    std::string source((std::istreambuf_iterator<char>(file)),{});
    nvrtcProgram program;
    RTC(nvrtcCreateProgram(&program,source.c_str(),"cuda_add.cu",0,nullptr,nullptr));
    const char* opts[]={"--gpu-architecture=compute_87","--std=c++17","--fmad=false"};
    auto rc=nvrtcCompileProgram(program,3,opts);
    if(rc!=NVRTC_SUCCESS) { size_t size; nvrtcGetProgramLogSize(program,&size); std::vector<char> log(size); nvrtcGetProgramLog(program,log.data()); fputs(log.data(),stderr); return 1; }
    size_t size; RTC(nvrtcGetPTXSize(program,&size)); std::vector<char> ptx(size);
    RTC(nvrtcGetPTX(program,ptx.data())); RTC(nvrtcDestroyProgram(&program));
    CUmodule module; CUfunction fn;
    DR(cuModuleLoadData(&module,ptx.data())); DR(cuModuleGetFunction(&fn,module,"add"));
    std::vector<float> a(n),b(n),out(n,0);
    for(size_t i=0;i<n;i++) { a[i]=float(i%97); b[i]=float(i%31); }
    float *da,*db,*dc;
    auto upload=Clock::now();
    RT(cudaMalloc(&da,bytes)); RT(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
    RT(cudaMalloc(&db,bytes)); RT(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    RT(cudaMalloc(&dc,bytes)); RT(cudaMemcpy(dc,out.data(),bytes,cudaMemcpyHostToDevice));
    RT(cudaDeviceSynchronize()); printf("upload_ms %.9f\n",ms(upload));
    void* args[]={&dc,&da,&db};
    auto launch=[&]() {
        cudaEvent_t start,end;
        if(timed) { RT(cudaEventCreate(&start)); RT(cudaEventCreate(&end)); RT(cudaEventRecord(start)); }
        DR(cuLaunchKernel(fn,rows,1,1,threads,1,1,0,nullptr,args,nullptr));
        if(timed) { RT(cudaEventRecord(end)); RT(cudaEventSynchronize(end)); float elapsed; RT(cudaEventElapsedTime(&elapsed,start,end)); fprintf(stderr,"kernel_ms %.9f\n",elapsed); RT(cudaEventDestroy(start)); RT(cudaEventDestroy(end)); }
        RT(cudaDeviceSynchronize());
    };
    for(int i=0;i<2000;i++) launch();
    auto host=Clock::now();
    for(int i=0;i<501;i++) launch();
    printf("host_us %.9f\n",ms(host)*1000/501);
    auto download=Clock::now();
    // Match Vx's new host output allocation on transfer-back.
    std::vector<float> answer(n);
    RT(cudaMemcpy(answer.data(),dc,bytes,cudaMemcpyDeviceToHost));
    RT(cudaDeviceSynchronize()); printf("download_ms %.9f\n",ms(download));
    for(size_t i=0;i<n;i++) if(answer[i]!=a[i]+b[i]) { fprintf(stderr,"FAIL at %zu\n",i); return 1; }
    puts("PASS");
    RT(cudaFree(da)); RT(cudaFree(db)); RT(cudaFree(dc)); DR(cuModuleUnload(module));
}
