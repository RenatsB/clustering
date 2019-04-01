#include <string>
#include <iostream>
#include "img.hpp"
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"
#include "gpuImageGen.h"
#include "gpuKmeans.h"
#include <chrono>

void outputImageData(std::string typePrefix,
                     const size_t x,
                     const size_t y,
                     const size_t numElements,
                     ColorVector &cv,
                     ImageColors &ic,
                     std::vector<float> &ln,
                     std::vector<float> &svR,
                     std::vector<float> &svG,
                     std::vector<float> &svB,
                     std::vector<float> &svA,
                     std::vector<float> &lvR,
                     std::vector<float> &lvG,
                     std::vector<float> &lvB,
                     std::vector<float> &lvA);

int main(int argc, char* argv[])
{
    ImageGenFn gen;
    cpuKmeans km;
    RandomFn<float> rg;
    size_t x = 512;
    size_t y = 512;
    size_t noiseSize = 128;
    size_t numIter = 2;
    size_t numClusters = 4;
    size_t numThreads = 32;

    ColorVector source_serial_CV =gen.generate_serial_CV(x,y,noiseSize);
    ImageColors source_serial_IC =gen.generate_serial_IC(x,y,noiseSize);
    std::vector<float> source_serial_LN = gen.generate_serial_LN(x,y,noiseSize);
    std::vector<float> source_serial_4SVr(x*y);
    std::vector<float> source_serial_4SVg(x*y);
    std::vector<float> source_serial_4SVb(x*y);
    std::vector<float> source_serial_4SVa(x*y);
    gen.generate_serial_4SV(x,y,noiseSize,
                            &source_serial_4SVr,
                            &source_serial_4SVg,
                            &source_serial_4SVb,
                            &source_serial_4SVa);
    std::vector<float> source_serial_4LVr(x*y);
    std::vector<float> source_serial_4LVg(x*y);
    std::vector<float> source_serial_4LVb(x*y);
    std::vector<float> source_serial_4LVa(x*y);
    gen.generate_serial_4LV(x,y,noiseSize,
                            source_serial_4LVr.data(),
                            source_serial_4LVg.data(),
                            source_serial_4LVb.data(),
                            source_serial_4LVa.data());


    ColorVector source_parallel_CV =gpuImageGen::generate_parallel_CV(x,y,
                                                                      noiseSize,
                                                                      numThreads);
    ImageColors source_parallel_IC =gpuImageGen::generate_parallel_IC(x,y,
                                                                      noiseSize,
                                                                      numThreads);
    std::vector<float> source_parallel_LN =gpuImageGen::generate_parallel_LN(x,y,
                                                                             noiseSize,
                                                                             numThreads);
    std::vector<float> source_parallel_4SVr(x*y);
    std::vector<float> source_parallel_4SVg(x*y);
    std::vector<float> source_parallel_4SVb(x*y);
    std::vector<float> source_parallel_4SVa(x*y);
    gpuImageGen::generate_parallel_4SV(&source_parallel_4SVr,
                                       &source_parallel_4SVg,
                                       &source_parallel_4SVb,
                                       &source_parallel_4SVa,
                                       x,y,noiseSize,numThreads);
    std::vector<float> source_parallel_4LVr(x*y);
    std::vector<float> source_parallel_4LVg(x*y);
    std::vector<float> source_parallel_4LVb(x*y);
    std::vector<float> source_parallel_4LVa(x*y);
    gpuImageGen::generate_parallel_4LV(source_parallel_4LVr.data(),
                                       source_parallel_4LVg.data(),
                                       source_parallel_4LVb.data(),
                                       source_parallel_4LVa.data(),
                                       x,y,noiseSize,numThreads);


    ColorVector filter_serial_CV = km.kmeans_serial_CV(source_serial_CV,
                                                              numClusters,
                                                              numIter);
    ImageColors filter_serial_IC = km.kmeans_serial_IC(source_serial_IC,
                                                              numClusters,
                                                              numIter);
    std::vector<float> filter_serial_LN = km.kmeans_serial_LN(source_serial_LN,
                                                              numClusters,
                                                              numIter);
    std::vector<float> filter_serial_4SVr(x*y);
    std::vector<float> filter_serial_4SVg(x*y);
    std::vector<float> filter_serial_4SVb(x*y);
    std::vector<float> filter_serial_4SVa(x*y);
    km.kmeans_serial_4SV(&source_serial_4SVr,
                         &source_serial_4SVg,
                         &source_serial_4SVb,
                         &source_serial_4SVa,
                         &filter_serial_4SVr,
                         &filter_serial_4SVg,
                         &filter_serial_4SVb,
                         &filter_serial_4SVa,
                         x*y,
                         numClusters,
                         numIter);
    std::vector<float> filter_serial_4LVr(x*y);
    std::vector<float> filter_serial_4LVg(x*y);
    std::vector<float> filter_serial_4LVb(x*y);
    std::vector<float> filter_serial_4LVa(x*y);
    km.kmeans_serial_4LV(source_serial_4LVr.data(),
                         source_serial_4LVg.data(),
                         source_serial_4LVb.data(),
                         source_serial_4LVa.data(),
                         filter_serial_4LVr.data(),
                         filter_serial_4LVg.data(),
                         filter_serial_4LVb.data(),
                         filter_serial_4LVa.data(),
                         x*y,
                         numClusters,
                         numIter);




    ColorVector filter_parallel_CV =
            gpuKmeans::kmeans_parallel_CV(source_serial_CV,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    ImageColors filter_parallel_IC =
            gpuKmeans::kmeans_parallel_IC(source_serial_IC,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    std::vector<float> filter_parallel_LN =
            gpuKmeans::kmeans_parallel_LN(source_serial_LN,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    std::vector<float> filter_parallel_4SVr(x*y);
    std::vector<float> filter_parallel_4SVg(x*y);
    std::vector<float> filter_parallel_4SVb(x*y);
    std::vector<float> filter_parallel_4SVa(x*y);
    gpuKmeans::kmeans_parallel_4SV(&source_parallel_4SVr,
                                   &source_parallel_4SVg,
                                   &source_parallel_4SVb,
                                   &source_parallel_4SVa,
                                   &filter_parallel_4SVr,
                                   &filter_parallel_4SVg,
                                   &filter_parallel_4SVb,
                                   &filter_parallel_4SVa,
                                   x*y,
                                   numClusters,
                                   numIter,
                                   numThreads,
                                   &rg);
    std::vector<float> filter_parallel_4LVr(x*y);
    std::vector<float> filter_parallel_4LVg(x*y);
    std::vector<float> filter_parallel_4LVb(x*y);
    std::vector<float> filter_parallel_4LVa(x*y);
    gpuKmeans::kmeans_parallel_4LV(source_parallel_4LVr.data(),
                                   source_parallel_4LVg.data(),
                                   source_parallel_4LVb.data(),
                                   source_parallel_4LVa.data(),
                                   filter_parallel_4LVr.data(),
                                   filter_parallel_4LVg.data(),
                                   filter_parallel_4LVb.data(),
                                   filter_parallel_4LVa.data(),
                                   x*y,
                                   numClusters,
                                   numIter,
                                   numThreads,
                                   &rg);

    outputImageData("generator_serial", x, y, x*y, source_serial_CV, source_serial_IC, source_serial_LN,
                    source_serial_4SVr,source_serial_4SVg,source_serial_4SVb,source_serial_4SVa,
                    source_serial_4LVr,source_serial_4LVg,source_serial_4LVb,source_serial_4LVa);

    outputImageData("generator_parallel", x, y, x*y, source_parallel_CV, source_parallel_IC, source_parallel_LN,
                    source_parallel_4SVr,source_parallel_4SVg,source_parallel_4SVb,source_parallel_4SVa,
                    source_parallel_4LVr,source_parallel_4LVg,source_parallel_4LVb,source_parallel_4LVa);

    outputImageData("filter_serial", x, y, x*y, filter_serial_CV, filter_serial_IC, filter_serial_LN,
                    filter_serial_4SVr,filter_serial_4SVg,filter_serial_4SVb,filter_serial_4SVa,
                    filter_serial_4LVr,filter_serial_4LVg,filter_serial_4LVb,filter_serial_4LVa);

    outputImageData("filter_parallel", x, y, x*y, filter_parallel_CV, filter_parallel_IC, filter_parallel_LN,
                    filter_parallel_4SVr,filter_parallel_4SVg,filter_parallel_4SVb,filter_parallel_4SVa,
                    filter_parallel_4LVr,filter_parallel_4LVg,filter_parallel_4LVb,filter_parallel_4LVa);
    return 0;
}

void outputImageData(std::string typePrefix,
                     const size_t x,
                     const size_t y,
                     const size_t numElements,
                     ColorVector &cv,
                     ImageColors &ic,
                     std::vector<float> &ln,
                     std::vector<float> &svR,
                     std::vector<float> &svG,
                     std::vector<float> &svB,
                     std::vector<float> &svA,
                     std::vector<float> &lvR,
                     std::vector<float> &lvG,
                     std::vector<float> &lvB,
                     std::vector<float> &lvA)
{
    std::vector<float> outp(numElements*4);
    for(uint i=0; i<numElements; ++i)
    {
        outp.at(i*4)   = cv.at(i).m_r;
        outp.at(i*4+1) = cv.at(i).m_g;
        outp.at(i*4+2) = cv.at(i).m_b;
        outp.at(i*4+3) = cv.at(i).m_a;
    }
    writeImage(typePrefix+"_CV.png",outp,x,y);

    for(uint i=0; i<numElements; ++i)
    {
        outp.at(i*4)   = ic.m_r.at(i);
        outp.at(i*4+1) = ic.m_g.at(i);
        outp.at(i*4+2) = ic.m_b.at(i);
        outp.at(i*4+3) = ic.m_a.at(i);
    }
    writeImage(typePrefix+"_IC.png",outp,x,y);

    std::copy(ln.begin(),ln.end(),outp.begin());
    writeImage(typePrefix+"_LN.png",outp,x,y);

    for(uint i=0; i<x*y; ++i)
    {
        outp.at(i*4)   = svR.at(i);
        outp.at(i*4+1) = svG.at(i);
        outp.at(i*4+2) = svB.at(i);
        outp.at(i*4+3) = svA.at(i);
    }
    writeImage(typePrefix+"_4SV.png",outp,x,y);

    for(uint i=0; i<x*y; ++i)
    {
        outp.at(i*4)   = lvR[i];
        outp.at(i*4+1) = lvG[i];
        outp.at(i*4+2) = lvB[i];
        outp.at(i*4+3) = lvA[i];
    }
    writeImage(typePrefix+"_4LV.png",outp,x,y);
    return;
}
