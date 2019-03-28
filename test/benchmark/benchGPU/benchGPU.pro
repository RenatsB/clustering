include($${PWD}/../../../common.pri)

TEMPLATE = app
TARGET = clib_GPU_benchmarks.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

LIBS += -L../../../clib -lclibp
QMAKE_RPATHDIR += ../../../clib

#LIBS += -lgtest -lgmock

HEADERS += $$files(include/*.h)
SOURCES += $$files(src/*.cpp)
CUDA_SOURCES += $$files(src/*.cu)

INCLUDEPATH += $$PWD/include
INCLUDEPATH += $$PWD/../include
INCLUDEPATH += $$PWD/../../../clib/clibs/include
INCLUDEPATH += $$PWD/../../../clib/clibp/include

LIBS += -lbenchmark

include($${PWD}/../../../cuda_compiler.pri)
