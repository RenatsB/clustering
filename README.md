# Image based K-Means Clustering

--

#### Naming conventions

Function names in this library have certain naming conventions for the ease of use.

Image generation and kmeans clustering functions use the following naming structure:
```
<name>_<type>_<processing type>
```
* <name> refers to actual name of the method, without any additional types.
* <type> refers to CPU/GPU or in other words serial/parallel
* <processing type> refers to data structure the method performs operations on.

```
<processing type>:
CV - ColorVector (vector of Color)
IC - ImageColors (Structure of float vectors)
LN - Linear (single vector of floats)
4SV - 4 Std::Vectors of floats
4LV - 4 pointers to arrays (or std::vectors)
```


--



references:

https://lodev.org/cgtutor/randomnoise.html
https://github.com/albelax/TexGen/blob/master/src/Image.cpp
https://lodev.org/cgtutor/index.html
http://www.cplusplus.com/reference/random/uniform_real_distribution/
https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
https://devtalk.nvidia.com/default/topic/1014332/jetson-tx2/data-sharing-between-c-and-cuda-programs/
https://github.com/OpenImageIO/oiio/blob/release/src/doc/openimageio.pdf
https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp
http://aresio.blogspot.com/2011/05/cuda-random-numbers-inside-kernels.html
https://github.com/albelax/StableFluids/blob/master/SolverGpu/cudasrc/rand_gpu.cu
https://nidclip.wordpress.com/2014/04/02/cuda-random-number-generation/
https://en.wikipedia.org/wiki/K-means%2B%2B
https://www.clear.rice.edu/comp422/resources/cuda/html/curand/device-api-overview.html#distributions
