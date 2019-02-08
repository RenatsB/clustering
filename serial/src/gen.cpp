#include <gen.h>
#include <chrono>

float Gen::generate(uint w, uint h)
{
    rawData.resize(h);
    for(auto c : rawData)
    {
        c.resize(w);
        for(auto t : c)
            t.resize(4);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();



    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;

    image = QImage((int)w,(int)h,QImage::Format_RGB16);

    return duration;
}
