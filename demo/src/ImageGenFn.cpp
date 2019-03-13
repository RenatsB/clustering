#include "ImageGenFn.hpp"

DataFrame ImageGenFn::generate(uint w, uint h, uint size)
{
    m_noiseWidth=w;
    m_noiseHeight=h;
    DataFrame rawData(w*h);
    m_noise.resize(h);
    for (auto &vert : m_noise )
        vert.resize(w);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.0,1.0);
    //generate the per-pixel noise
    generateNoise();

    //Create a variable here to avoid reallocation in the loop
    Color pixelColor;

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, size);
            float pwr2 = turbulence(x, y+m_noiseHeight, size/2);
            float pwr3 = turbulence(x, y+m_noiseHeight*2, size/2);

            pixelColor.setData(pwr1,pwr2,pwr3);

            rawData.at(y*w+x).setData(pixelColor.m_r,pixelColor.m_g,pixelColor.m_b);
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
   value  = fractX       * fractY       * m_noise.at(y1).at(x1);
   value += (1 - fractX) * fractY       * m_noise.at(y1).at(x2);
   value += fractX       * (1 - fractY) * m_noise.at(y2).at(x1);
   value += (1 - fractX) * (1 - fractY) * m_noise.at(y2).at(x2);

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
  m_noise.resize(m_noiseHeight*3);
  for(auto &l : m_noise)
      l.resize(m_noiseWidth);
  for (uint y = 0; y < m_noise.size(); y++)
    for (uint x = 0; x < m_noiseWidth; x++)
    {
        m_noise.at(y).at(x) = m_rand.MT19937RandU();
    }
}
