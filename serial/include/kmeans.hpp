#ifndef CLUSTERING_KMEANS_HPP_
#define CLUSTERING_KMEANS_HPP_
#include <vector>
#include "ran.hpp"
/*struct Point {
  double x{0}, y{0};
};

struct Colour
{
    double r{0}, g{0}, b{0}, a{1.0};
};

using DataFrame = std::vector<Colour>;*/

class kmeans
{
public:
std::vector<double> k_means(const std::vector<double>& data,
                           size_t format,
                           size_t k,
                           size_t number_of_iterations);
private:
double square(double value);
//double squared_l2_distance(Point first, Point second);
double squared_Colour_l2_Distance(double r1, double g1, double b1, double a1,
                                  double r2, double g2, double b2, double a2);
private:
Ran rfunc;
};

#endif //CLUSTERING_KMEANS_HPP_
