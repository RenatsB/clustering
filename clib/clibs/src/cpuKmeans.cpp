#include "cpuKmeans.hpp"
#include <random>
#include <iostream>

float cpuKmeans::square(float value) {
  return value * value;
}

float cpuKmeans::squared_Colour_l2_Distance(Color first, Color second)
{
  return square(first.m_r*first.m_a - second.m_r*second.m_a)
       + square(first.m_g*first.m_a - second.m_g*second.m_a)
       + square(first.m_b*first.m_a - second.m_b*second.m_a);
}

float cpuKmeans::linear_squared_Colour_l2_Distance(float FR,float FG,float FB,float FA,
                                         float SR,float SG,float SB,float SA)
{
  return square(FR*FA - SR*SA) + square(FG*FA - SG*SA) + square(FB*FA - SB*SA);
}

ColorVector cpuKmeans::kmeans_serial_CV(const ColorVector& data,
                          size_t k,
                          size_t number_of_iterations) {
    rfunc.setNumericLimitsL(0, data.size() - 1);

    ColorVector correctedImage(data.size());
    ColorVector means(k);
    // Pick centroids as random points from the dataset.
    /*for (auto& cluster : means) {
      cluster = data[rfunc.MT19937RandL()];
    }*/

    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    std::vector<float> distances(data.size());
    std::vector<size_t> weights(data.size(),0);
    means[0] = data[number]; //first mean is random
    for(auto centroid=1; centroid<k; ++centroid)
    {
        for(auto p=0; p<data.size(); ++p)
        {
            for(auto c=0; c<centroid; ++c)
            {
                distances.at(p)=squared_Colour_l2_Distance(data.at(p), means.at(c));
                weights.at(p)+=(size_t)(distances.at(p)*1000.f);
            }
        }
        size_t randomIDx = rfunc.weightedRand(weights);
        means[centroid]=data.at(randomIDx);
    }
    //end of centoid picking



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
      ColorVector new_means(k);
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

ImageColors cpuKmeans::kmeans_serial_IC(const ImageColors& data,
                                        size_t k,
                                        size_t number_of_iterations)
{
    size_t numberOfItems = data.getSize();
    rfunc.setNumericLimitsL(0, numberOfItems - 1);

    ImageColors correctedImage;
    correctedImage.resize(numberOfItems);
    ImageColors means;
    means.resize(k);
    // Pick centroids as random points from the dataset.
    /*for (auto cluster=0; cluster<k; ++cluster) {
      size_t num = rfunc.MT19937RandL();
      means.setData(cluster,
                    data.m_r.at(num),
                    data.m_g.at(num),
                    data.m_b.at(num),
                    data.m_a.at(num));
    }*/

    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    std::vector<float> distances(numberOfItems);
    std::vector<size_t> weights(numberOfItems,0);
    //first mean is random
    means.m_r.at(0) = data.m_r.at(number);
    means.m_g.at(0) = data.m_g.at(number);
    means.m_b.at(0) = data.m_b.at(number);
    means.m_a.at(0) = data.m_a.at(number);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        for(auto p=0; p<numberOfItems; ++p)
        {
            for(auto c=0; c<centroid; ++c)
            {
                distances.at(p)=linear_squared_Colour_l2_Distance(
                            data.m_r.at(p),
                            data.m_g.at(p),
                            data.m_b.at(p),
                            data.m_a.at(p),
                            means.m_r.at(c),
                            means.m_g.at(c),
                            means.m_b.at(c),
                            means.m_a.at(c));
                weights.at(p)+=(size_t)(distances.at(p)*1000.f);
            }
        }
        size_t randomIDx = rfunc.weightedRand(weights);
        means.m_r.at(centroid)=data.m_r.at(randomIDx);
        means.m_g.at(centroid)=data.m_g.at(randomIDx);
        means.m_b.at(centroid)=data.m_b.at(randomIDx);
        means.m_a.at(centroid)=data.m_a.at(randomIDx);
    }
    //end of centoid picking


    std::vector<size_t> assignments(numberOfItems);
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      // Find assignments.
      for (size_t point = 0; point < numberOfItems; ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance =
              linear_squared_Colour_l2_Distance(data.m_r.at(point),data.m_g.at(point),data.m_b.at(point), data.m_a.at(point),
                                            means.m_r.at(cluster),means.m_g.at(cluster),means.m_b.at(cluster), means.m_a.at(cluster));
          if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        assignments[point] = best_cluster;
      }

      // Sum up and count points for each cluster.
      ImageColors new_means;
      new_means.resize(k);
      std::vector<size_t> counts(k, 0);
      for (size_t point = 0; point < numberOfItems; ++point) {
        const auto cluster = assignments[point];
        new_means.m_r.at(cluster) += data.m_r.at(point);
        new_means.m_g.at(cluster) += data.m_g.at(point);
        new_means.m_b.at(cluster) += data.m_b.at(point);
        new_means.m_a.at(cluster) += data.m_a.at(point);
        counts[cluster] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster) {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        means.m_r.at(cluster) = new_means.m_r.at(cluster) / count;
        means.m_g.at(cluster) = new_means.m_g.at(cluster) / count;
        means.m_b.at(cluster) = new_means.m_b.at(cluster) / count;
        means.m_a.at(cluster) = new_means.m_a.at(cluster) / count;
      }
    }

    for(uint i=0; i<numberOfItems; ++i)
    {
        correctedImage.m_r.at(i) = means.m_r.at(assignments[i]);
        correctedImage.m_g.at(i) = means.m_g.at(assignments[i]);
        correctedImage.m_b.at(i) = means.m_b.at(assignments[i]);
        correctedImage.m_a.at(i) = means.m_a.at(assignments[i]);
    }


    return correctedImage;
}

std::vector<float> cpuKmeans::kmeans_serial_LN(const std::vector<float>& data,
                          size_t k,
                          size_t number_of_iterations) {
    const size_t dataSize = data.size()/4;
    rfunc.setNumericLimitsL(0, dataSize - 1);

    std::vector<float> correctedImage(data.size());
    std::vector<float> means(k*4);
    // Pick centroids as random points from the dataset.
    /*for (auto& cluster : means) {
      cluster = data[rfunc.MT19937RandL()];
    }*/



    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    std::vector<float> distances(dataSize);
    std::vector<size_t> weights(dataSize,0);
    means[0] = data[number*4]; //first mean is random
    means[1] = data[number*4+1];
    means[2] = data[number*4+2];
    means[3] = data[number*4+3];
    for(auto centroid=1; centroid<k; ++centroid)
    {
        for(auto p=0; p<dataSize; ++p)
        {
            for(auto c=0; c<centroid; ++c)
            {
                distances.at(p)=linear_squared_Colour_l2_Distance(
                            data.at(p*4),
                            data.at(p*4+1),
                            data.at(p*4+2),
                            data.at(p*4+3),
                            means.at(c*4),
                            means.at(c*4+1),
                            means.at(c*4+2),
                            means.at(c*4+3));
                weights.at(p)+=(size_t)(distances.at(p)*1000.f);
            }
        }
        size_t randomIDx = rfunc.weightedRand(weights);
        means[centroid*4]  =data.at(randomIDx*4);
        means[centroid*4+1]=data.at(randomIDx*4+1);
        means[centroid*4+2]=data.at(randomIDx*4+2);
        means[centroid*4+3]=data.at(randomIDx*4+3);
    }
    //end of centoid picking



    std::vector<size_t> assignments(dataSize);
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      // Find assignments.
      for (size_t point = 0; point < dataSize; ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance = linear_squared_Colour_l2_Distance(
                      data[point*4],
                      data[point*4+1],
                      data[point*4+2],
                      data[point*4+3],
                      means[cluster*4],
                      means[cluster*4+1],
                      means[cluster*4+2],
                      means[cluster*4+3]);
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
      for (size_t point = 0; point < dataSize; ++point) {
        const auto cluster = assignments[point];
        new_means[cluster*4]   += data[point*4];
        new_means[cluster*4+1] += data[point*4+1];
        new_means[cluster*4+2] += data[point*4+2];
        new_means[cluster*4+3] += data[point*4+3];
        counts[cluster] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster) {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        means[cluster*4]   = new_means[cluster*4] / count;
        means[cluster*4+1] = new_means[cluster*4+1] / count;
        means[cluster*4+2] = new_means[cluster*4+2] / count;
        means[cluster*4+3] = new_means[cluster*4+3] / count;
      }
    }

    for(uint i=0; i<dataSize; ++i)
    {
        correctedImage.at(i*4)   = means[assignments[i]*4];
        correctedImage.at(i*4+1) = means[assignments[i]*4+1];
        correctedImage.at(i*4+2) = means[assignments[i]*4+2];
        correctedImage.at(i*4+3) = means[assignments[i]*4+3];
    }
    return correctedImage;
}

void cpuKmeans::kmeans_serial_4SV(const std::vector<float>* _inreds,
                                  const std::vector<float>* _ingrns,
                                  const std::vector<float>* _inblus,
                                  const std::vector<float>* _inalps,
                                  std::vector<float>* _outreds,
                                  std::vector<float>* _outgrns,
                                  std::vector<float>* _outblus,
                                  std::vector<float>* _outalps,
                                  const size_t num_items,
                                  size_t k,
                                  size_t number_of_iterations)
{
    rfunc.setNumericLimitsL(0, num_items - 1);
    std::vector<float> meansR(k);
    std::vector<float> meansG(k);
    std::vector<float> meansB(k);
    std::vector<float> meansA(k);
    // Pick centroids as random points from the dataset.
    /*for (uint cluster=0; cluster<k; ++cluster) {
      size_t num = rfunc.MT19937RandL();
      meansR.at(cluster) = _inreds->at(num);
      meansG.at(cluster) = _ingrns->at(num);
      meansB.at(cluster) = _inblus->at(num);
      meansA.at(cluster) = _inalps->at(num);
    }*/


    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    std::vector<float> distances(num_items);
    std::vector<size_t> weights(num_items,0);
    //first mean is random
    meansR.at(0) = _inreds->at(number);
    meansG.at(0) = _ingrns->at(number);
    meansB.at(0) = _inblus->at(number);
    meansA.at(0) = _inalps->at(number);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        for(auto p=0; p<num_items; ++p)
        {
            for(auto c=0; c<centroid; ++c)
            {
                distances.at(p)=linear_squared_Colour_l2_Distance(
                            _inreds->at(p),
                            _ingrns->at(p),
                            _inblus->at(p),
                            _inalps->at(p),
                            meansR.at(c),
                            meansG.at(c),
                            meansB.at(c),
                            meansA.at(c));
                weights.at(p)+=(size_t)(distances.at(p)*1000.f);
            }
        }
        size_t randomIDx = rfunc.weightedRand(weights);
        meansR.at(centroid)=_inreds->at(randomIDx);
        meansG.at(centroid)=_ingrns->at(randomIDx);
        meansB.at(centroid)=_inblus->at(randomIDx);
        meansA.at(centroid)=_inalps->at(randomIDx);
    }
    //end of centoid picking


    std::vector<size_t> assignments(num_items);
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
      // Find assignments.
      for (size_t point = 0; point < num_items; ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance =
              linear_squared_Colour_l2_Distance(_inreds->at(point),
                                                _ingrns->at(point),
                                                _inblus->at(point),
                                                _inalps->at(point),
                                                meansR.at(cluster),
                                                meansG.at(cluster),
                                                meansB.at(cluster),
                                                meansA.at(cluster));
          if (distance < best_distance)
          {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        assignments[point] = best_cluster;
      }

      // Sum up and count points for each cluster.
      std::vector<float> new_meansR(k);
      std::vector<float> new_meansG(k);
      std::vector<float> new_meansB(k);
      std::vector<float> new_meansA(k);
      std::vector<size_t> counts(k, 0);
      for (size_t point = 0; point < num_items; ++point)
      {
        new_meansR.at(assignments.at(point)) += _inreds->at(point);
        new_meansG.at(assignments.at(point)) += _ingrns->at(point);
        new_meansB.at(assignments.at(point)) += _inblus->at(point);
        new_meansA.at(assignments.at(point)) += _inalps->at(point);
        counts[assignments[point]] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster)
      {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        meansR[cluster] = new_meansR[cluster] / count;
        meansG[cluster] = new_meansG[cluster] / count;
        meansB[cluster] = new_meansB[cluster] / count;
        meansA[cluster] = new_meansA[cluster] / count;
      }
    }

    for(uint i=0; i<num_items; ++i)
    {
        _outreds->at(i) = meansR.at(assignments[i]);
        _outgrns->at(i) = meansG.at(assignments[i]);
        _outblus->at(i) = meansB.at(assignments[i]);
        _outalps->at(i) = meansA.at(assignments[i]);
    }
    return;
}

void cpuKmeans::kmeans_serial_4LV(const float* _inreds,
                                  const float* _ingrns,
                                  const float* _inblus,
                                  const float* _inalps,
                                  float* _outreds,
                                  float* _outgrns,
                                  float* _outblus,
                                  float* _outalps,
                                  const size_t num_items,
                                  size_t k,
                                  size_t number_of_iterations)
{
    rfunc.setNumericLimitsL(0, num_items - 1);
    std::vector<float> meansR(k);
    std::vector<float> meansG(k);
    std::vector<float> meansB(k);
    std::vector<float> meansA(k);
    // Pick centroids as random points from the dataset.
    /*for (uint cluster=0; cluster<k; ++cluster)
    {
      size_t num = rfunc.MT19937RandL();
      meansR[cluster] = _inreds[num];
      meansG[cluster] = _ingrns[num];
      meansB[cluster] = _inblus[num];
      meansA[cluster] = _inalps[num];
    }*/


    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    std::vector<float> distances(num_items);
    std::vector<size_t> weights(num_items,0);
    //first mean is random
    meansR.at(0) = _inreds[number];
    meansG.at(0) = _ingrns[number];
    meansB.at(0) = _inblus[number];
    meansA.at(0) = _inalps[number];
    for(auto centroid=1; centroid<k; ++centroid)
    {
        for(auto p=0; p<num_items; ++p)
        {
            for(auto c=0; c<centroid; ++c)
            {
                distances.at(p)=linear_squared_Colour_l2_Distance(
                            _inreds[p],
                            _ingrns[p],
                            _inblus[p],
                            _inalps[p],
                            meansR.at(c),
                            meansG.at(c),
                            meansB.at(c),
                            meansA.at(c));
                weights.at(p)+=(size_t)(distances.at(p)*1000.f);
            }
        }
        size_t randomIDx = rfunc.weightedRand(weights);
        meansR.at(centroid)=_inreds[randomIDx];
        meansG.at(centroid)=_ingrns[randomIDx];
        meansB.at(centroid)=_inblus[randomIDx];
        meansA.at(centroid)=_inalps[randomIDx];
    }
    //end of centoid picking


    std::vector<size_t> assignments(num_items);
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration)
    {
      // Find assignments.
      for (size_t point = 0; point < num_items; ++point) {
        float best_distance = std::numeric_limits<float>::max();
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
          const float distance =
              linear_squared_Colour_l2_Distance(_inreds[point],
                                                _ingrns[point],
                                                _inblus[point],
                                                _inalps[point],
                                                meansR[cluster],
                                                meansG[cluster],
                                                meansB[cluster],
                                                meansA[cluster]);
          if (distance < best_distance)
          {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        assignments[point] = best_cluster;
      }

      // Sum up and count points for each cluster.
      std::vector<float> new_meansR(k);
      std::vector<float> new_meansG(k);
      std::vector<float> new_meansB(k);
      std::vector<float> new_meansA(k);
      std::vector<size_t> counts(k, 0);
      for (size_t point = 0; point < num_items; ++point)
      {
        new_meansR[assignments[point]] += _inreds[point];
        new_meansG[assignments[point]] += _ingrns[point];
        new_meansB[assignments[point]] += _inblus[point];
        new_meansA[assignments[point]] += _inalps[point];
        counts[assignments[point]] += 1;
      }

      // Divide sums by counts to get new centroids.
      for (size_t cluster = 0; cluster < k; ++cluster)
      {
        // Turn 0/0 into 0/1 to avoid zero division.
        const auto count = std::max<size_t>(1, counts[cluster]);
        meansR[cluster] = new_meansR[cluster] / count;
        meansG[cluster] = new_meansG[cluster] / count;
        meansB[cluster] = new_meansB[cluster] / count;
        meansA[cluster] = new_meansA[cluster] / count;
      }
    }

    for(uint i=0; i<num_items; ++i)
    {
        _outreds[i] = meansR[assignments[i]];
        _outgrns[i] = meansG[assignments[i]];
        _outblus[i] = meansB[assignments[i]];
        _outalps[i] = meansA[assignments[i]];
    }
    return;
}
