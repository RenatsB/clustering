#include <string>
#include <iostream>
#include "img.hpp"
#include "ImageGenFn.hpp"
#include "kmeans.hpp"
#include "kmeansP.h"
#include "ImageGen.h"
#include <chrono>

int main(int argc, char* argv[])
{
    ImageGenFn m_gen;
    RandomFn<float> rfunc;
    uint x = 2048;
    uint y = 2048;
    uint noiseSize = 128;
    uint numIter = 1;
    uint numClusters = 6;
    uint numThreads = 1024;

    //look at half float, struct of arrays
    //get rid of ss seen in compiler explorer
    //https://www.godbolt.org/

    const char* nameGCS = "GeneratorColorSerial.jpg";
    const char* nameCSF = "SerialColorSerialFiltered.jpg";
    const char* nameCPF = "SerialColorParallelFiltered.jpg";
    const char* nameGLS = "GeneratorLinearSerial.jpg";
    const char* nameLSF = "SerialLinearSerialFiltered.jpg";
    const char* nameLPF = "SerialLinearParallelFiltered.jpg";
    const char* nameGCP = "GeneratorColorParallel.jpg";
    const char* nameGLP = "GeneratorLinearParallel.jpg";
    kmeans k;

    float gens1Duration =0.f;
    float gens2Duration =0.f;
    float gens3Duration =0.f;
    float gens4Duration =0.f;
    float gens5Duration =0.f;
    float gens6Duration =0.f;
    float genp1Duration =0.f;
    float genp2Duration =0.f;
    float flt1sDuration =0.f;
    float flt1pDuration =0.f;
    float flt2sDuration =0.f;
    float flt2pDuration =0.f;
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
    ColorVector parallelSource = generate(x,y, noiseSize, numThreads);
    t2 = std::chrono::high_resolution_clock::now();
    genp1Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color structure version generation finished on device in "<<genp1Duration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> parallelLinearSource = linear_generate(x,y, noiseSize, numThreads);
    t2 = std::chrono::high_resolution_clock::now();
    genp2Duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version generation finished on device in "<<genp2Duration<<" seconds..."<<std::endl;




    t1 = std::chrono::high_resolution_clock::now();
    ColorVector serialOut = k.k_means(source,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    flt1sDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on host in "<<flt1sDuration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linFiltered = k.linear_k_means(linearSource,numClusters,numIter);
    t2 = std::chrono::high_resolution_clock::now();
    flt2sDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Linear 1D version filtered on host in "<<flt2sDuration<<" seconds..."<<std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> parallelOut = kmeansP(source,numClusters,numIter,&rfunc,1024);
    t2 = std::chrono::high_resolution_clock::now();
    flt1pDuration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    std::cout<<"Color version filtered on device in "<<flt1pDuration<<" seconds..."<<std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> linParallelOut = linear_kmeansP(linearSource,numClusters,numIter,&rfunc,numThreads);
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
}
