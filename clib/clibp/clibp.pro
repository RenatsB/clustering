include($${PWD}/../../common.pri)
TEMPLATE = lib
TARGET = clibp
DESTDIR = ../

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

INCLUDEPATH+=$$PWD/include
INCLUDEPATH+=$$PWD/../clibs/include

HEADERS += $$files($$PWD/include/*.cuh)
CUDA_SOURCES += $$files($$PWD/src/*.cu)

include($${PWD}/../../cuda_compiler.pri)
