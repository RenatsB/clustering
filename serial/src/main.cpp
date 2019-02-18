#include <string>
#include <iostream>
#include "gen.h"

int main(int argc, char* argv[])
{
    Gen m_gen;
    float time1 = m_gen.generate(512,512,true);
    std::cout<<"Serial version generated image in "<<time1<<"seconds..."<<std::endl;
    return 0;
}
