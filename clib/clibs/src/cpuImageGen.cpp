#include "cpuImageGen.hpp"

ColorVector ImageGenFn::generate_serial_CV(const uint w,
                               const uint h,
                               const uint turbulence_size)
{
    m_noiseWidth=w;
    m_noiseHeight=h;
    ColorVector rawData(w*h);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+h, turbulence_size/2);
            float pwr3 = turbulence(x, y+h*2, turbulence_size/2);

            rawData.at(y*w+x).setData(pwr1,pwr2,pwr3);
        }
    }
    //end of map generation
    return rawData;
}

ImageColors ImageGenFn::generate_serial_IC(const uint w,
                   const uint h,
                   const uint turbulence_size)
{
    m_noiseWidth=w;
    m_noiseHeight=h;
    ImageColors data;
    data.resize(w*h);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+h, turbulence_size/2);
            float pwr3 = turbulence(x, y+h*2, turbulence_size/2);
            data.setData(y*w+x, pwr1, pwr2, pwr3, 1.f);
        }
    }
    //end of map generation
    return data;
}

std::vector<float> ImageGenFn::generate_serial_LN(const uint w,
                               const uint h,
                               const uint turbulence_size)
{
    m_noiseWidth=w;
    m_noiseHeight=h;
    std::vector<float> rawData(w*h*4);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+h, turbulence_size/2);
            float pwr3 = turbulence(x, y+h*2, turbulence_size/2);

            rawData.at((y*w+x)*4)=pwr1;
            rawData.at((y*w+x)*4+1)=pwr2;
            rawData.at((y*w+x)*4+2)=pwr3;
            rawData.at((y*w+x)*4+3)=1.0f;
        }
    }
    //end of map generation
    return rawData;
}

void ImageGenFn::generate_serial_4SV(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   std::vector<float>* redChannel,
                   std::vector<float>* greenChannel,
                   std::vector<float>* blueChannel,
                   std::vector<float>* alphaChannel)
{
    m_noiseWidth=w;
    m_noiseHeight=h;

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+h, turbulence_size/2);
            float pwr3 = turbulence(x, y+h*2, turbulence_size/2);

            redChannel->at(y*w+x)=pwr1;
            greenChannel->at(y*w+x)=pwr2;
            blueChannel->at(y*w+x)=pwr3;
            alphaChannel->at(y*w+x)=1.0f;
        }
    }
    //end of map generation
}

void ImageGenFn::generate_serial_4LV(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   float* redChannel,
                   float* greenChannel,
                   float* blueChannel,
                   float* alphaChannel)
{
    m_noiseWidth=w;
    m_noiseHeight=h;

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            float pwr1 = turbulence(x, y, turbulence_size);
            float pwr2 = turbulence(x, y+h, turbulence_size/2);
            float pwr3 = turbulence(x, y+h*2, turbulence_size/2);

            redChannel[y*w+x]=pwr1;
            greenChannel[y*w+x]=pwr2;
            blueChannel[y*w+x]=pwr3;
            alphaChannel[y*w+x]=1.0f;
        }
    }
    //end of map generation
}

void ImageGenFn::generate_serial_4LL(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   float* redChannel,
                   float* greenChannel,
                   float* blueChannel,
                   float* alphaChannel)
{
    m_noiseWidth=w;
    m_noiseHeight=h;

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.00000f,1.00000f);
    //generate the per-pixel noise
    generateNoise();

    //generate the map here
    for(size_t y=0; y<h; ++y)
    {
        for(size_t x=0; x<w; ++x)
        {
            redChannel[y*w+x]=turbulence(x, y, turbulence_size);
            greenChannel[y*w+x]=turbulence(x, y+h, turbulence_size/2);
            blueChannel[y*w+x]=turbulence(x, y+h*2, turbulence_size/2);
            alphaChannel[y*w+x]=turbulence(x, h-y, turbulence_size*2);
        }
    }
    //end of map generation
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
