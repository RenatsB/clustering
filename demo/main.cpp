#include <string>
#include <iostream>
#include "img.hpp"
#include "ImageGenFn.hpp"
#include "kmeans.hpp"
#include "kmeansP.cuh"
#include <chrono>

int main(int argc, char* argv[])
{
    ImageGenFn m_gen;
    RandomFn<float> rfunc;
    uint x = 2048;
    uint y = 2048;
    uint numIter = 1;
    uint numClusters = 8;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    DataFrame source = m_gen.generate(x,y, 128, 1024, 1024);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color structure version generation finished in "<<duration<<"seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linearSource = m_gen.linear_generate(x,y, 128, 1024, 1024);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version generation finished in "<<duration<<"seconds..."<<std::endl;

    const char* nameA1 = "BaseSource.jpg";
    const char* nameB1 = "SerialFiltered.jpg";
    const char* nameB2 = "ParallelFiltered.jpg";
    const char* nameL1 = "LinearSource.jpg";
    const char* nameL2 = "LinearSerialFiltered.jpg";
    const char* nameL3 = "LinearParallelFiltered.jpg";
    std::vector<float> outp(source.size()*4);
    kmeans k;
    t1 = std::chrono::high_resolution_clock::now();
    DataFrame serialOut = k.k_means(source,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on host in "<<duration<<"seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linFiltered = k.linear_k_means(linearSource,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version filtered on host in "<<duration<<"seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> parallelOut = kmeansP(source,numClusters,numIter,&rfunc);
    t2 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on device in "<<duration<<"seconds..."<<std::endl;
    //outp.resize(source.size()*4);
    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4)   = source.at(i).m_r;
        outp.at(i*4+1) = source.at(i).m_g;
        outp.at(i*4+2) = source.at(i).m_b;
        outp.at(i*4+3) = source.at(i).m_a;
    }


    writeImage(nameA1,outp,x,y);
    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4) = serialOut.at(i).m_r;
        outp.at(i*4+1) = serialOut.at(i).m_g;
        outp.at(i*4+2) = serialOut.at(i).m_b;
        outp.at(i*4+3) = serialOut.at(i).m_a;
    }
    writeImage(nameB1,outp,x,y);
    for(uint i=0; i<outp.size(); ++i)
    {
        outp.at(i) = parallelOut.at(i);
    }
    writeImage(nameB2,outp,x,y);

    outp = linearSource;
    writeImage(nameL1,outp,x,y);

    outp = linFiltered;
    writeImage(nameL2,outp,x,y);
    return 0;
}
