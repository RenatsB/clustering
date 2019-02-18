#include "img.h"

img::~img()
[

]

bool img::openFile(std::string filename)
{
    if(image!=nullptr)
        closeFile();
    ImageInput *in = ImageInput::open (filename);
    if (! in)
        return false;
    const ImageSpec &spec = in->spec();
    int xres = spec.width;
    int yres = spec.height;
    int channels = spec.nchannels;
    std::vector<unsigned char> pixels (xres*yres*channels);
    in->read_image (TypeDesc::UINT8, &pixels[0]);

}

bool img::closeFile()
{
    in->close ();
    ImageInput::destroy (in);
}
