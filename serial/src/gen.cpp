#include <gen.hpp>

DataFrame Gen::generate(uint w, uint h)
{
    DataFrame rawData(w*h);
    m_noise.resize(h);
    for (auto vert : m_noise )
        vert.resize(w);

    //this has to be set before a number within desired limits can be acquired
    m_rand.setNumericLimits(0.0,1.0);
    //generate the per-pixel noise
    generateNoise();

    //set limits to 0-256 for colors
    m_rand.setNumericLimits(0.0,256.0);

    //generate 4 different random colors
    Color L1,L2,L3,L4;
    L1.setData(m_rand.MT19937RandU(),
               m_rand.MT19937RandU(),
               m_rand.MT19937RandU());
    L2.setData(m_rand.MT19937RandU(),
               m_rand.MT19937RandU(),
               m_rand.MT19937RandU());
    L3.setData(m_rand.MT19937RandU(),
               m_rand.MT19937RandU(),
               m_rand.MT19937RandU());
    L4.setData(m_rand.MT19937RandU(),
               m_rand.MT19937RandU(),
               m_rand.MT19937RandU());

    //Create a variable here to avoid reallocation in the loop
    Color pixelColor;

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x; x<w; ++x)
        {
            double pwr = 192 + turbulence(x, y, 64) / 4;
            pixelColor.setData(pwr*L1.m_r,pwr*L1.m_g,pwr*L1.m_b);
        }
    }
    //end of map generation

    return rawData;
}

double Gen::smoothNoise(double x, double y)
{
   //get fractional part of x and y
   double fractX = x - int(x);
   double fractY = y - int(y);

   //wrap around
   int x1 = (int(x) + m_noiseWidth) % m_noiseWidth;
   int y1 = (int(y) + m_noiseHeight) % m_noiseHeight;

   //neighbor values
   int x2 = (x1 + m_noiseWidth - 1) % m_noiseWidth;
   int y2 = (y1 + m_noiseHeight - 1) % m_noiseHeight;

   //smooth the noise with bilinear interpolation
   double value;
   value  = fractX       * fractY       * m_noise.at(y1).at(x1);
   value += (1 - fractX) * fractY       * m_noise.at(y1).at(x2);
   value += fractX       * (1 - fractY) * m_noise.at(y2).at(x1);
   value += (1 - fractX) * (1 - fractY) * m_noise.at(y2).at(x2);

   return value;
}

double Gen::turbulence(double x, double y, double size)
{
  double value = 0.0, initialSize = size;

  while(size >= 1)
  {
    value += smoothNoise(x / size, y / size) * size;
    size /= 2.0;
  }

  return(128.0 * value / initialSize);
}

void Gen::generateNoise()
{
  for (int y = 0; y < m_noiseHeight; y++)
  for (int x = 0; x < m_noiseWidth; x++)
  {
    m_noise.at(y).at(x) = m_rand.MT19937RandU();
  }
}
