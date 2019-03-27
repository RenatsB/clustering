include($${PWD}/../../common.pri)

TEMPLATE = app
TARGET = clibGPUtests.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

INCLUDEPATH+= include /usr/local/include /public/devel/include

LIBS+= -L/usr/local/lib -lgtest -lpthread \
       -L/public/devel/lib/ -lgtest \
       -L../../clib -lclibp

INCLUDEPATH+=../../clib/clibs/include
INCLUDEPATH+=../../clib/clibp/include
INCLUDEPATH+=$$PWD/../include

QMAKE_RPATHDIR += ../../clib


SOURCES += $$files($$PWD/src/*.cpp)
SOURCES += $$files($$PWD/../src/testUtils.cpp)
CUDA_SOURCES += $$files($$PWD/src/*.cu)

HEADERS += $$PWD/include/*.h
HEADERS += $$PWD/../include/testUtils.h

include($${PWD}/../../cuda_compiler.pri)
