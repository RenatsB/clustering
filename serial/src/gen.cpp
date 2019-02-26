#include <gen.hpp>
#include <chrono>
#include <iostream>

std::vector<double> Gen::generate(uint w, uint h, uint format)
{
    rawData.resize(w*h*format);

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x; x<w; ++x)
        {
            rawData.at(y*w+x) = m_rand.randd(0.0,256.0,1);
            rawData.at(y*w+x+1) = m_rand.randd(0.0,256.0,1);
            rawData.at(y*w+x+2) = m_rand.randd(0.0,256.0,1);
            rawData.at(y*w+x+3) = 1.0;
        }
    }
    for(auto &c : rawData)
    {
        c = m_rand.randi(0,256,1);
    }
    //end of map generation

    return rawData;
}
