#ifndef CLUSTERING_GEN_HPP_
#define CLUSTERING_GEN_HPP_

#include <vector>
#include <QImage>
#include <string>
#include "ran.hpp"

class Gen
{
public:
    std::vector<double> generate(uint w, uint h, uint format=3);
private:
    std::vector<double> rawData;
    Ran m_rand;
};

#endif //CLUSTERING_GEN_HPP_
