#ifndef CLUSTERING_KMEANS_HPP_
#define CLUSTERING_KMEANS_HPP_
#include "utilTypes.hpp"
#include "ran.hpp"

class kmeans
{
public:
DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations);
private:
double square(double value);
//double squared_l2_distance(Color first, Color second);
double squared_Colour_l2_Distance(Color first, Color second);
private:
Ran<double> rfunc;
};

#endif //CLUSTERING_KMEANS_HPP_
