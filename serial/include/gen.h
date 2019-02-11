#ifndef GEN_GEN_H_
#define GEN_GEN_H_

#include <vector>
#include <QImage>
#include <string>
#include "ran.h"

class Gen
{
public:
    float generate(uint w, uint h, bool _output=false);
private:
    void outputImage();

    std::vector<std::vector<std::vector<uint>>> rawData;
    Ran m_rand;
};

#endif //GEN_GEN_H_
