#ifndef CLUSTERING_IMG_H_
#define CLUSTERING_IMG_H_
#include <OpenImageIO/imageio.h>
#include <string>
OIIO_NAMESPACE_USING
class img
{
public:
    img()=default;
    ~img();
    bool openFile(std::string filename);
    bool closeFile();
private:
    ImageInput* image = nullptr;
};

#endif //CLUSTERING_IMG_H_
