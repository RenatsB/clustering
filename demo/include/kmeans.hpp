#ifndef CLUSTERING_KMEANS_HPP_
#define CLUSTERING_KMEANS_HPP_
#include "utilTypes.hpp"
#include "RandomFn.hpp"

class kmeans
{
public:
DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations);
std::vector<float> linear_k_means(const std::vector<float>& data,
                  size_t k,
                  size_t number_of_iterations);
private:
float square(float value);
//double squared_l2_distance(Color first, Color second);
float squared_Colour_l2_Distance(Color first, Color second);
float linear_squared_Colour_l2_Distance(float FR,float FG,float FB,
                                        float SR,float SG,float SB);
private:
RandomFn<float> rfunc;
};

#endif //CLUSTERING_KMEANS_HPP_