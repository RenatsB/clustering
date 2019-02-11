#include <gen.h>
#include <chrono>

float Gen::generate(uint w, uint h, bool _output)
{
    rawData.resize(h);
    for(auto c : rawData)
    {
        c.resize(w);
        for(auto t : c)
            t.resize(4);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x; x<w; ++x)
        {
            rawData.at(y).at(x).at(0) = m_rand.randi(0,256,1);
            rawData.at(y).at(x).at(1) = m_rand.randi(0,256,1);
            rawData.at(y).at(x).at(2) = m_rand.randi(0,256,1);
            rawData.at(y).at(x).at(3) = 256;
        }
    }
    //end of map generation

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    float duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / 1000000.f;

    if(_output)
        outputImage();

    return duration;
}

void Gen::outputImage()
{
    QImage image = QImage((int)rawData.size(),(int)rawData.at(0).size(),QImage::Format_RGB16);
    QColor col;
    for(uint y=0; y<rawData.size(); ++y)
    {
        for(uint x; x<rawData.at(0).size(); ++x)
        {
            col.setRgb(rawData.at(y).at(x).at(0),rawData.at(y).at(x).at(1),rawData.at(y).at(x).at(2),rawData.at(y).at(x).at(3));
            image.setPixelColor(x,y,col);
        }
    }

    image.save("../NoiseGen_Serial_OUT.jpg");
}
