include($${PWD}/../common.pri)
TEMPLATE = app
TARGET = KMeansClustering
#DESTDIR = build

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

INCLUDEPATH+=$$PWD/include
INCLUDEPATH+=$$PWD/../clib/clibs/include
INCLUDEPATH+=$$PWD/../clib/clibp/include

HEADERS += $$files($$PWD/img.hpp)
SOURCES += $$files($$PWD/main.cpp)

LIBS += -L../clib -lclibs -lclibp
LIBS += -lOpenImageIO

QMAKE_RPATHDIR += ../clib

include($${PWD}/../cuda_compiler.pri)
