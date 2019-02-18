#include "img.h"

img::~img()
{
    closeFile();
}

bool img::openFile(std::string filename)
{
    if(image!=nullptr)
        closeFile();
    image = ImageInput::open (filename);
    if (! image)
        return false;
    const ImageSpec &spec = image->spec();
    int xres = spec.width;
    int yres = spec.height;
    int channels = spec.nchannels;
    std::vector<unsigned char> pixels (xres*yres*channels);
    image->read_image (TypeDesc::UINT8, &pixels[0]);
    return true;
}

void img::closeFile()
{
    image->close ();
    ImageInput::destroy (image.get());
}
