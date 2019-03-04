#include <string>
#include <iostream>
#include "img.hpp"
#include "gen.hpp"
#include <chrono>

int main(int argc, char* argv[])
{
    Gen m_gen;

    uint x = 512;
    uint y = 512;
    DataFrame ret = m_gen.generate(x,y);
    const char* name = "Test.jpg";

    std::vector<double> outp;
    outp.resize(ret.size()*4);
    for(uint i=0; i<ret.size(); ++i)
    {
        outp.at(i) = ret.at(i).m_r;
        outp.at(i+1) = ret.at(i).m_g;
        outp.at(i+2) = ret.at(i).m_b;
        outp.at(i+3) = ret.at(i).m_a;
    }
    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //float time1 = m_gen.generate(512,512,true);
    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    //std::cout<<"Serial version finished in "<<time1<<"seconds..."<<std::endl;
    writeImage(name,outp,x,y);
    return 0;
}
