#include <string>
#include <iostream>
#include "gen.hpp"
#include <chrono>

int main(int argc, char* argv[])
{
    Gen m_gen;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    //float time1 = m_gen.generate(512,512,true);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;
    //std::cout<<"Serial version finished in "<<time1<<"seconds..."<<std::endl;
    return 0;
}
