#ifndef CLUSTERING_GEN_H_
#define CLUSTERING_GEN_H_

#include <vector>
#include <QImage>
#include <string>
#include "ran.h"

class Gen
{
public:
    float generate(uint w, uint h, uint format=3, bool _output=false);
private:
    void outputImage();

    std::vector<uint> rawData;
    Ran m_rand;
};

#endif //CLUSTERING_GEN_H_
