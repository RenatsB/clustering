include($${PWD}/../../../common.pri)

TEMPLATE = app
TARGET = clib_CPU_benchmarks.out

OBJECTS_DIR = obj
CUDA_OBJECTS_DIR = cudaobj

LIBS += -L../../../clib -lclibs
QMAKE_RPATHDIR += ../../../clib

#LIBS += -lgtest -lgmock

HEADERS += $$files(include/*.hpp)
SOURCES += $$files(src/*.cpp)

INCLUDEPATH += $$PWD/include
INCLUDEPATH += $$PWD/../include
INCLUDEPATH += $$PWD/../../../clib/clibs/include

LIBS += -lbenchmark
