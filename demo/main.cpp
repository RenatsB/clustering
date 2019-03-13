#include <string>
#include <iostream>
#include "img.hpp"
#include "ImageGenFn.hpp"
#include "kmeans.hpp"
#include "kmeansP.cuh"
//#include <chrono>

int main(int argc, char* argv[])
{
    ImageGenFn m_gen;
    RandomFn<float> rfunc;
    uint x = 8024;
    uint y = 8024;
    uint numIter = 1;
    uint numClusters = 8;
    DataFrame source = m_gen.generate(x,y, 128);
    const char* nameA1 = "BaseSource.jpg";
    const char* nameB1 = "SerialFiltered.jpg";
    const char* nameB2 = "ParallelFiltered.jpg";
    std::vector<float> outp(source.size()*4);
    kmeans k;
    DataFrame serialOut = k.k_means(source,numClusters,numIter);
    std::vector<float> parallelOut = kmeansP(source,numClusters,numIter,&rfunc);
    //outp.resize(source.size()*4);
    for(uint i=0; i<source.size(); ++i)
    {
        outp.at(i*4)   = source.at(i).m_r;
        outp.at(i*4+1) = source.at(i).m_g;
        outp.at(i*4+2) = source.at(i).m_b;
        outp.at(i*4+3) = source.at(i).m_a;
    }
    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //float time1 = m_gen.generate(512,512,true);
    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    //std::cout<<"Serial version finished in "<<time1<<"seconds..."<<std::endl;
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
    return 0;
}
