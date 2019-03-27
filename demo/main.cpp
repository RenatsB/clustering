#include <string>
#include <iostream>
#include "img.hpp"
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"
#include "gpuImageGen.h"
#include "gpuKmeans.h"
#include <chrono>

int main(int argc, char* argv[])
{
    ImageGenFn gen;
    cpuKmeans km;
    RandomFn<float> rg;
    uint x = 64;
    uint y = 64;
    uint noiseSize = 64;
    uint numIter = 1;
    uint numClusters = 4;
    uint numThreads = 64;

    const char* nameGsCV = "GeneratorSerialCV.png";
    const char* nameGsIC = "GeneratorSerialIC.png";
    const char* nameGsLN = "GeneratorSerialLN.png";
    const char* nameGs4SV = "GeneratorSerial4SV.png";
    const char* nameGs4LV = "GeneratorSerial4LN.png";

    const char* nameGpCV = "GeneratorParallelCV.png";
    const char* nameGpIC = "GeneratorParallelIC.png";
    const char* nameGpLN = "GeneratorParallelLN.png";
    const char* nameGp4SV = "GeneratorParallel4SV.png";
    const char* nameGp4LV = "GeneratorParallel4LN.png";

    const char* nameFsCV = "FilterSerialCV.png";
    const char* nameFsIC = "FilterSerialIC.png";
    const char* nameFsLN = "FilterSerialLN.png";
    const char* nameFs4SV = "FilterSerial4SV.png";
    const char* nameFs4LV = "FilterSerial4LV.png";

    const char* nameFpCV = "FilterParallelCV.png";
    const char* nameFpIC = "FilterParallelIC.png";
    const char* nameFpLN = "FilterParallelLN.png";
    const char* nameFp4SV = "FilterParallel4SV.png";
    const char* nameFp4LV = "FilterParallel4LV.png";

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
    gpuImageGen::generate_parallel_4SV(x,y,noiseSize,
                                       &source_parallel_4SVr,
                                       &source_parallel_4SVg,
                                       &source_parallel_4SVb,
                                       &source_parallel_4SVa,
                                       numThreads);
    std::vector<float> source_parallel_4LVr(x*y);
    std::vector<float> source_parallel_4LVg(x*y);
    std::vector<float> source_parallel_4LVb(x*y);
    std::vector<float> source_parallel_4LVa(x*y);
    gpuImageGen::generate_parallel_4SV(x,y,noiseSize,
                                       source_parallel_4LVr.data(),
                                       source_parallel_4LVg.data(),
                                       source_parallel_4LVb.data(),
                                       source_parallel_4LVa.data(),
                                       numThreads);


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
    km.kmeans_serial_4LV(&source_serial_4LVr,
                         &source_serial_4LVg,
                         &source_serial_4LVb,
                         &source_serial_4LVa,
                         &filter_serial_4LVr,
                         &filter_serial_4LVg,
                         &filter_serial_4LVb,
                         &filter_serial_4LVa,
                         x*y,
                         numClusters,
                         numIter);




    ColorVector filter_parallel_CV =
            gpuKmeans::kmeans_parallel_CV(source_serial_CV,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    ImageColors filter_serial_IC =
            gpuKmeans::kmeans_parallel_IC(source_serial_IC,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    std::vector<float> filter_serial_LN =
            gpuKmeans::kmeans_parallel_LN(source_serial_LN,
                                          numClusters,
                                          numIter,
                                          numThreads,&rg);
    std::vector<float> filter_serial_4SVr(x*y);
    std::vector<float> filter_serial_4SVg(x*y);
    std::vector<float> filter_serial_4SVb(x*y);
    std::vector<float> filter_serial_4SVa(x*y);
    gpuKmeans::kmeans_parallel_4SV(&source_serial_4SVr,
                                   &source_serial_4SVg,
                                   &source_serial_4SVb,
                                   &source_serial_4SVa,
                                   &filter_serial_4SVr,
                                   &filter_serial_4SVg,
                                   &filter_serial_4SVb,
                                   &filter_serial_4SVa,
                                   numClusters,
                                   numIter,
                                   numThreads,
                                   &rg);
    std::vector<float> filter_serial_4LVr(x*y);
    std::vector<float> filter_serial_4LVg(x*y);
    std::vector<float> filter_serial_4LVb(x*y);
    std::vector<float> filter_serial_4LVa(x*y);
    gpuKmeans::kmeans_parallel_4LV(&source_serial_4LVr,
                                   &source_serial_4LVg,
                                   &source_serial_4LVb,
                                   &source_serial_4LVa,
                                   &filter_serial_4LVr,
                                   &filter_serial_4LVg,
                                   &filter_serial_4LVb,
                                   &filter_serial_4LVa,
                                   numClusters,
                                   numIter,
                                   numThreads,
                                   &rg);


    std::vector<float> outp(source_serial_CV.size()*4);
    for(uint i=0; i<source_serial_CV.size(); ++i)
    {
        outp.at(i*4)   = source_serial_CV.at(i).m_r;
        outp.at(i*4+1) = source_serial_CV.at(i).m_g;
        outp.at(i*4+2) = source_serial_CV.at(i).m_b;
        outp.at(i*4+3) = source_serial_CV.at(i).m_a;
    }
    writeImage(nameGsCV,outp,x,y);

    for(uint i=0; i<source_serial_CV.size(); ++i)
    {
        outp.at(i*4)   = source_serial_IC.m_r.at(i);
        outp.at(i*4+1) = source_serial_IC.m_g.at(i);
        outp.at(i*4+2) = source_serial_IC.m_b.at(i);
        outp.at(i*4+3) = source_serial_IC.m_a.at(i);
    }
    writeImage(nameGsIC,outp,x,y);

    std::copy(source_serial_LN.begin(),source_serial_LN.end(),outp.begin());
    writeImage(nameGsLN,outp,x,y);

    for(uint i=0; i<x*y; ++i)
    {
        outp.at(i*4)   = source_serial_4SVr.at(i);
        outp.at(i*4+1) = source_serial_4SVg.at(i);
        outp.at(i*4+2) = source_serial_4SVb.at(i);
        outp.at(i*4+3) = source_serial_4SVa.at(i);
    }
    writeImage(nameGs4SV,outp,x,y);

    for(uint i=0; i<x*y; ++i)
    {
        outp.at(i*4)   = source_serial_4LVr.at(i);
        outp.at(i*4+1) = source_serial_4LVg.at(i);
        outp.at(i*4+2) = source_serial_4LVb.at(i);
        outp.at(i*4+3) = source_serial_4LVa.at(i);
    }
    writeImage(nameGs4LV,outp,x,y);

    return 0;
}


/*
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    ColorVector source = m_gen.generate_serial_CV(x,y, noiseSize);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    gens1Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color structure version generation finished on host in "<<gens1Duration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linearSource = m_gen.generate_serial_LN(x,y, noiseSize);
    t2 = std::chrono::high_resolution_clock::now();
    gens2Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version generation finished on host in "<<gens2Duration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    ImageColors imgColSource = m_gen.generate_serial_IC(x,y, noiseSize);
    t2 = std::chrono::high_resolution_clock::now();
    gens3Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"ImageColor structure version generation finished on host in "<<gens3Duration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> redChan(x*y);
    std::vector<float> grnChan(x*y);
    std::vector<float> bluChan(x*y);
    std::vector<float> alpChan(x*y);
    m_gen.generate_serial_4SV(x,y, noiseSize, &redChan, &grnChan, &bluChan, &alpChan);
    t2 = std::chrono::high_resolution_clock::now();
    gens4Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"4 std::vector<float> version generation finished on host in "<<gens4Duration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> BredChan(x*y);
    std::vector<float> BgrnChan(x*y);
    std::vector<float> BbluChan(x*y);
    std::vector<float> BalpChan(x*y);
    m_gen.generate_serial_4LV(x,y, noiseSize, BredChan.data(), BgrnChan.data(), BbluChan.data(), BalpChan.data());
    t2 = std::chrono::high_resolution_clock::now();
    gens5Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"4 float* version generation finished on host in "<<gens5Duration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> AredChan(x*y);
    std::vector<float> AgrnChan(x*y);
    std::vector<float> AbluChan(x*y);
    std::vector<float> AalpChan(x*y);
    m_gen.generate_serial_4LL(x,y, noiseSize, AredChan.data(), AgrnChan.data(), AbluChan.data(), AalpChan.data());
    t2 = std::chrono::high_resolution_clock::now();
    gens6Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"4 float* version generation finished on host in "<<gens6Duration<<" seconds..."<<std::endl;


    t1 = std::chrono::high_resolution_clock::now();
    ColorVector parallelSource = gpuImageGen::generate_parallel_CV(x,y, noiseSize, numThreads);
    t2 = std::chrono::high_resolution_clock::now();
    genp1Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color structure version generation finished on device in "<<genp1Duration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> parallelLinearSource = gpuImageGen::generate_parallel_LN(x,y, noiseSize, numThreads);
    t2 = std::chrono::high_resolution_clock::now();
    genp2Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version generation finished on device in "<<genp2Duration<<" seconds..."<<std::endl;




    t1 = std::chrono::high_resolution_clock::now();
    ColorVector serialOut = k.kmeans_serial_CV(source,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    flt1sDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on host in "<<flt1sDuration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linFiltered = k.kmeans_serial_LN(linearSource,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    flt2sDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version filtered on host in "<<flt2sDuration<<" seconds..."<<std::endl;

    ImageColors cerialKmIC = k.kmeans_serial_IC(imgColSource,numClusters,numIter);


    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> parallelOut = gpuKmeans::kmeans_parallel_CV(source,numClusters,numIter,1024,&rfunc);
    t2 = std::chrono::high_resolution_clock::now();
    flt1pDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on device in "<<flt1pDuration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linParallelOut = gpuKmeans::kmeans_parallel_LN(linearSource,numClusters,numIter,numThreads,&rfunc);
    t2 = std::chrono::high_resolution_clock::now();
    flt2pDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version filtered on device in "<<flt2pDuration<<" seconds..."<<std::endl;

    std::cout<<"---------------------------------------"<<std::endl;

    std::cout<<"Linear is "<<gens1Duration/gens2Duration<<" times quicker than Color structure based on generation"<<std::endl;

    std::cout<<"Host filter is "<<flt1sDuration/flt1pDuration<<" times quicker than Device filter using Color structure based"<<std::endl;
    std::cout<<"Host filter is "<<flt2sDuration/flt2pDuration<<" times quicker than Device filter using linear"<<std::endl;

    std::cout<<"Linear is "<<flt1sDuration/flt2sDuration<<" times quicker than Color structure based on Host"<<std::endl;
    std::cout<<"Linear is "<<flt1pDuration/flt2pDuration<<" times quicker than Color structure based on Device"<<std::endl;

    std::cout<<"---------------------------------------"<<std::endl;

    std::vector<float> outp(source.size()*4);

    //outp.resize(source.size()*4);
    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4)   = source.at(i).m_r;
        outp.at(i*4+1) = source.at(i).m_g;
        outp.at(i*4+2) = source.at(i).m_b;
        outp.at(i*4+3) = source.at(i).m_a;
    }
    writeImage(nameGCS,outp,x,y);

    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4)   = cerialKmIC.m_r.at(i);
        outp.at(i*4+1) = cerialKmIC.m_g.at(i);
        outp.at(i*4+2) = cerialKmIC.m_b.at(i);
        outp.at(i*4+3) = cerialKmIC.m_a.at(i);
    }
    writeImage(nameGCS,outp,x,y);

    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4) = serialOut.at(i).m_r;
        outp.at(i*4+1) = serialOut.at(i).m_g;
        outp.at(i*4+2) = serialOut.at(i).m_b;
        outp.at(i*4+3) = serialOut.at(i).m_a;
    }
    writeImage(nameCSF,outp,x,y);


    for(uint i=0; i<parallelSource.size(); ++i)
    {
        outp.at(i*4) = parallelSource.at(i).m_r;
        outp.at(i*4+1) = parallelSource.at(i).m_g;
        outp.at(i*4+2) = parallelSource.at(i).m_b;
        outp.at(i*4+3) = parallelSource.at(i).m_a;
    }
    writeImage(nameGCP,outp,x,y);
    outp = parallelLinearSource;
    writeImage(nameGLP,outp,x,y);



    for(uint i=0; i<outp.size(); ++i)
    {
        outp.at(i) = parallelOut.at(i);
    }
    writeImage(nameCPF,outp,x,y);

    outp = linearSource;
    writeImage(nameGLS,outp,x,y);

    outp = linFiltered;
    writeImage(nameLSF,outp,x,y);

    outp = linParallelOut;
    writeImage(nameLPF,outp,x,y);
    return 0;
*/
