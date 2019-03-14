#include "kmeans.hpp"
#include <random>

float kmeans::square(float value) {
  return value * value;
}
/*
double kmeans::squared_l2_distance(Point first, Point second) {
  return square(first.x - second.x) + square(first.y - second.y);
}
*/
float kmeans::squared_Colour_l2_Distance(Color first, Color second)
{
  return square(first.m_r*first.m_a - second.m_r*second.m_a)
       + square(first.m_g*first.m_a - second.m_g*second.m_a)
       + square(first.m_b*first.m_a - second.m_b*second.m_a);
}

DataFrame kmeans::k_means(const DataFrame& data,
                          size_t k,
                          size_t number_of_iterations) {
    //static std::random_device seed;
    //static std::mt19937 random_number_generator(seed());
    //std::uniform_int_distribution<size_t> indices(0, data.size() - 1);
    rfunc.setNumericLimitsL(0, data.size() - 1);

    DataFrame correctedImage(data.size());

    // Pick centroids as random points from the dataset.
    DataFrame means(k);
    for (auto& cluster : means) {
      cluster = data[rfunc.MT19937RandL()];
    }

    std::vector<size_t> assignments(data.size());
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      // Find assignments.
      for (size_t point = 0; point < data.size(); ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance =
              squared_Colour_l2_Distance(data[point], means[cluster]);
          if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        assignments[point] = best_cluster;
      }

      // Sum up and count points for each cluster.
      DataFrame new_means(k);
      std::vector<size_t> counts(k, 0);
      for (size_t point = 0; point < data.size(); ++point) {
        const auto cluster = assignments[point];
        new_means[cluster].m_r += data[point].m_r;
        new_means[cluster].m_g += data[point].m_g;
        new_means[cluster].m_b += data[point].m_b;
        new_means[cluster].m_a += data[point].m_a;
        counts[cluster] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster) {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        means[cluster].m_r = new_means[cluster].m_r / count;
        means[cluster].m_g = new_means[cluster].m_g / count;
        means[cluster].m_b = new_means[cluster].m_b / count;
        means[cluster].m_a = new_means[cluster].m_a / count;
      }
    }

    for(uint i=0; i<correctedImage.size(); ++i)
    {
        correctedImage.at(i).setData(means[assignments[i]].m_r,
                                     means[assignments[i]].m_g,
                                     means[assignments[i]].m_b,
                                     means[assignments[i]].m_a);
    }

    return correctedImage;
}

//=======================================================================
//--------------------------|   LINEAR   |-------------------------------
//=======================================================================

float kmeans::linear_squared_Colour_l2_Distance(float FR,float FG,float FB,
                                         float SR,float SG,float SB)
{
  return square(FR - SR) + square(FG - SG) + square(FB - SB);
}

std::vector<float> kmeans::linear_k_means(const std::vector<float>& data,
                          size_t k,
                          size_t number_of_iterations) {
    size_t numberOfItems = data.size()/4;
    rfunc.setNumericLimitsL(0, numberOfItems - 1);

    std::vector<float> correctedImage(data.size());

    // Pick centroids as random points from the dataset.
    std::vector<float> means(k*4);

    for (uint cluster=0; cluster<k; ++cluster) {
      size_t num = rfunc.MT19937RandL();
      means[cluster*4]   = data[num*4];
      means[cluster*4+1] = data[num*4+1];
      means[cluster*4+2] = data[num*4+2];
      means[cluster*4+3] = data[num*4+3];
    }

    std::vector<size_t> assignments(numberOfItems);
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      // Find assignments.
      for (size_t point = 0; point < numberOfItems; ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance =
              linear_squared_Colour_l2_Distance(data[point*4],data[point*4+1],data[point*4+2],means[cluster*4],means[cluster*4+1],means[cluster*4+2]);
          if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        assignments[point] = best_cluster;
      }

      // Sum up and count points for each cluster.
      std::vector<float> new_means(k*4);
      std::vector<size_t> counts(k, 0);
      for (size_t point = 0; point < numberOfItems; ++point) {
        new_means[assignments[point]*4]   += data[point*4];
        new_means[assignments[point]*4+1] += data[point*4+1];
        new_means[assignments[point]*4+2] += data[point*4+2];
        new_means[assignments[point]*4+3] += data[point*4+3];
        counts[assignments[point]] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster) {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        means[cluster*4]   = new_means[cluster*4]   / count;
        means[cluster*4+1] = new_means[cluster*4+1] / count;
        means[cluster*4+2] = new_means[cluster*4+2] / count;
        means[cluster*4+3] = new_means[cluster*4+3] / count;
      }
    }

    for(uint i=0; i<numberOfItems; ++i)
    {
        correctedImage[i*4]   = means[assignments[i]*4];
        correctedImage[i*4+1] = means[assignments[i]*4+1];
        correctedImage[i*4+2] = means[assignments[i]*4+2];
        correctedImage[i*4+3] = means[assignments[i]*4+3];
    }

    return correctedImage;
}
