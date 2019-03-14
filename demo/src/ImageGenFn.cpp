#include "ImageGenFn.hpp"

std::vector<float> ImageGenFn::linear_generate(const uint w,
                               const uint h,
                               const uint turbulence_size,
                               const size_t noiseWidth,
                               const size_t noiseHeight)
{
    m_noiseWidth=noiseWidth;
    m_noiseHeight=noiseHeight;
    std::vector<float> rawData(w*h*4);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.0,1.0);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+m_noiseHeight, turbulence_size/2);
            float pwr3 = turbulence(x, y+m_noiseHeight*2, turbulence_size/2);

            rawData.at(y*w+x)=pwr1;
            rawData.at(y*w+x+1)=pwr2;
            rawData.at(y*w+x+2)=pwr3;
            rawData.at(y*w+x+3)=1.0f;
        }
    }
    //end of map generation
    return rawData;
}

DataFrame ImageGenFn::generate(const uint w,
                               const uint h,
                               const uint turbulence_size,
                               const size_t noiseWidth,
                               const size_t noiseHeight)
{
    m_noiseWidth=noiseWidth;
    m_noiseHeight=noiseHeight;
    DataFrame rawData(w*h);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.0,1.0);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+m_noiseHeight, turbulence_size/2);
            float pwr3 = turbulence(x, y+m_noiseHeight*2, turbulence_size/2);

            rawData.at(y*w+x).setData(pwr1,pwr2,pwr3);
        }
    }
    //end of map generation
    return rawData;
}


float ImageGenFn::smoothNoise(float x, float y)
{
   //get fractional part of x and y
   float fractX = x - int(x);
   float fractY = y - int(y);

   //wrap around
   int x1 = (int(x) + m_noiseWidth) % m_noiseWidth;
   int y1 = (int(y) + m_noiseHeight) % m_noiseHeight;

   //neighbor values
   int x2 = (x1 + m_noiseWidth - 1) % m_noiseWidth;
   int y2 = (y1 + m_noiseHeight - 1) % m_noiseHeight;

   //smooth the noise with bilinear interpolation
   float value;
   value  = fractX       * fractY       * m_noise.at(y1*m_noiseWidth+x1);
   value += (1 - fractX) * fractY       * m_noise.at(y1*m_noiseWidth+x2);
   value += fractX       * (1 - fractY) * m_noise.at(y2*m_noiseWidth+x1);
   value += (1 - fractX) * (1 - fractY) * m_noise.at(y2*m_noiseWidth+x2);

   return value;
}

float ImageGenFn::turbulence(float x, float y, float size)
{
  float value = 0.0, initialSize = size;

  while(size >= 1)
  {
    value += smoothNoise(x / size, y / size) * size;
    size /= 2.0;
  }

  return(128.0 * value / initialSize)/256.0;
}

void ImageGenFn::generateNoise()
{
  m_noise.resize(m_noiseHeight*m_noiseWidth);
  for (uint i = 0; i < m_noise.size(); ++i)
    m_noise.at(i) = m_rand.MT19937RandU();
}
