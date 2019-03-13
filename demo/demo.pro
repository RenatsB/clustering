include($${PWD}/../common.pri)
TEMPLATE = app
TARGET = KMeansClustering
DESTDIR = lib

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = parallel/cudaobj

INCLUDEPATH+=$$PWD/include

HEADERS += $$files($$PWD/include/*.hpp) \
           $$files($$PWD/include/*.cuh)
SOURCES += $$files($$PWD/src/*.cpp)
SOURCES += $$files($$PWD/main.cpp)
CUDA_SOURCES += $$files($$PWD/src/*.cu)

LIBS += -lOpenImageIO

include($${PWD}/../cuda_compiler.pri)
