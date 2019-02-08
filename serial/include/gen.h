#ifndef GEN_GEN_H_
#define GEN_GEN_H_

#include "ran.h"
#include <vector>
#include <QImage>
#include <string>
class Gen
{
public:
    Gen()=default;
    ~Gen()=default;
    float generate(uint w, uint h);
private:
    void outputImage();

    std::vector<std::vector<std::vector<uint>>> rawData;
    Ran m_rand;
};

#endif //GEN_GEN_H_
