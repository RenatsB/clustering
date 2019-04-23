#ifndef CLUSTERING_KMEANS_HPP_
#define CLUSTERING_KMEANS_HPP_
#include "utilTypes.hpp"
#include "cpuRandomFn.hpp"
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note CPU K-Means clustering solver
//-------------------------------------------------------------------------------------------------------
class cpuKmeans
{
public:
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ColorVector data type
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
//-----------------------------------------------------------------------------------------------------
ColorVector kmeans_serial_CV(const ColorVector& data,
                             size_t k,
                             size_t number_of_iterations);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ImageColors data type
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
//-----------------------------------------------------------------------------------------------------
ImageColors kmeans_serial_IC(const ImageColors& data,
                             size_t k,
                             size_t number_of_iterations);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using single vector of float values
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
//-----------------------------------------------------------------------------------------------------
std::vector<float> kmeans_serial_LN(const std::vector<float>& data,
                                    size_t k,
                                    size_t number_of_iterations);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ColorVector data type
/// @param [in]_inreds Source image red channel
/// @param [in]_ingrns Source image green channel
/// @param [in]_inblus Source image blue channel
/// @param [in]_inalps Source image alpha channel
/// @param [out]_outreds filtered image red output channel
/// @param [out]_outgrns filtered image green output channel
/// @param [out]_outblus filtered image blue output channel
/// @param [out]_outalps filtered image alpha output channel
/// @param [in]num_items number of pixels to process (= input and output vector sizes)
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
//-----------------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ColorVector data type
/// @param [in]_inreds Source image red channel
/// @param [in]_ingrns Source image green channel
/// @param [in]_inblus Source image blue channel
/// @param [in]_inalps Source image alpha channel
/// @param [out]_outreds filtered image red output channel
/// @param [out]_outgrns filtered image green output channel
/// @param [out]_outblus filtered image blue output channel
/// @param [out]_outalps filtered image alpha output channel
/// @param [in]num_items number of pixels to process (= input and output vector sizes)
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
//-----------------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------------
/// @brief Convenience method for squaring
/// @param [in]value a value to square
//-----------------------------------------------------------------------------------------------------
float square(float value);
//-----------------------------------------------------------------------------------------------------
/// @brief Gets the distance between two vectors. Alpha is used a a weight. 
/// @brief Square root skipped for performance reasons.
/// @param [in]first first color vector
/// @param [in]second second color vector
//-----------------------------------------------------------------------------------------------------
float squared_Colour_l2_Distance(Color first, Color second);
//-----------------------------------------------------------------------------------------------------
/// @brief Gets the distance between two vectors. Alpha is used a a weight. 
/// @brief Square root skipped for performance reasons.
/// @param [in]FR first color red value
/// @param [in]FG first color green value
/// @param [in]FB first color blue value
/// @param [in]FA first color alpha value
/// @param [in]SR second color red value
/// @param [in]SG second color green value
/// @param [in]SB second color blue value
/// @param [in]SA second color alpha value
//-----------------------------------------------------------------------------------------------------
float linear_squared_Colour_l2_Distance(float FR,float FG,float FB,float FA,
                                        float SR,float SG,float SB, float SA);
private:
//-----------------------------------------------------------------------------------------------------
/// @brief Random number generator wrapper.
//-----------------------------------------------------------------------------------------------------
RandomFn<float> rfunc;
};

#endif //CLUSTERING_KMEANS_HPP_
