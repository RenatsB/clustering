#include <gen.hpp>

DataFrame Gen::generate(uint w, uint h, uint size)
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

    //set limits to 0-256 for colors
    m_rand.setNumericLimits(0.0,1.0);

    //generate 4 different random colors
    Color L1,L2,L3,L4;
    //L1.setData(256.0,0.0,0.0,0.0);
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
    Color pixelColor, a1,a2,a3;

    //generate the map here
    for(uint y=0; y<h; ++y)
    {
        for(uint x=0; x<w; ++x)
        {
            double pwr1 = turbulence(x, y, size);


            /*double pwr2 = turbulence(x, y, size/2);
            a1.setData(pwr2*L1.m_r,pwr2*L1.m_g,pwr2*L1.m_b);

            double pwr3 = turbulence(x, y, size/4);
            a2.setData(pwr3*L1.m_r,pwr3*L1.m_g,pwr3*L1.m_b);

            double pwr4 = turbulence(x, y, size/8);
            a3.setData(pwr4*L1.m_r,pwr4*L1.m_g,pwr4*L1.m_b);*/

            /*pixelColor.m_r+=a1.m_r*pwr2;
            pixelColor.m_g+=a1.m_g*pwr2;
            pixelColor.m_b+=a1.m_b*pwr2;*/

            double pwr2 = turbulence(x, y+m_noiseHeight, size/2);
            double pwr3 = turbulence(x, y+m_noiseHeight*2, size/2);

            pixelColor.setData(pwr1,pwr2,pwr3);

            rawData.at(y*w+x).setData(pixelColor.m_r,pixelColor.m_g,pixelColor.m_b);
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

  return(128.0 * value / initialSize)/256.0;
}

void Gen::generateNoise()
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
