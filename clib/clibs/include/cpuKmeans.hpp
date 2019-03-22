#ifndef CLUSTERING_KMEANS_HPP_
#define CLUSTERING_KMEANS_HPP_
#include "utilTypes.hpp"
#include "cpuRandomFn.hpp"

class cpuKmeans
{
public:
ColorVector kmeans_serial_CV(const ColorVector& data,
                             size_t k,
                             size_t number_of_iterations);
ImageColors kmeans_serial_IC(const ImageColors& data,
                             size_t k,
                             size_t number_of_iterations);
std::vector<float> kmeans_serial_LN(const std::vector<float>& data,
                                    size_t k,
                                    size_t number_of_iterations);
void kmeans_serial_4SV(const std::vector<float>* _inreds,
                       const std::vector<float>* _ingrns,
                       const std::vector<float>* _inblus,
                       const std::vector<float>* _inalps,
                       std::vector<float>* _outreds,
                       std::vector<float>* _outgrns,
                       std::vector<float>* _outblus,
                       std::vector<float>* _outalps,
                       const size_t num_items,
                       size_t k,
                       size_t number_of_iterations);
void kmeans_serial_4LV(const float* _inreds,
                       const float* _ingrns,
                       const float* _inblus,
                       const float* _inalps,
                       float* _outreds,
                       float* _outgrns,
                       float* _outblus,
                       float* _outalps,
                       const size_t num_items,
                       size_t k,
                       size_t number_of_iterations);
private:
float square(float value);
float squared_Colour_l2_Distance(Color first, Color second);
float linear_squared_Colour_l2_Distance(float FR,float FG,float FB,
                                        float SR,float SG,float SB);
private:
RandomFn<float> rfunc;
};

#endif //CLUSTERING_KMEANS_HPP_
