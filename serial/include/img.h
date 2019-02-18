#ifndef CLUSTERING_IMG_H_
#define CLUSTERING_IMG_H_
#include <OpenImageIO/imageio.h>
#include <string>
#include <memory>
OIIO_NAMESPACE_USING
class img
{
public:
    img()=default;
    ~img();
    bool openFile(std::string filename);
    void closeFile();
private:
    std::unique_ptr<ImageInput> image = nullptr;
};

#endif //CLUSTERING_IMG_H_
