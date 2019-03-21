include($${PWD}/../../common.pri)

TARGET = clibCPUtests.out

OBJECTS_DIR = obj

INCLUDEPATH+= include /usr/local/include /public/devel/include

LIBS+= -L/usr/local/lib -lgtest -lpthread \
       -L/public/devel/lib/ -lgtest \
       -L../../clib -lclibs -lclibp

INCLUDEPATH+=../../clib/clibs/include
INCLUDEPATH+=../../clib/clibp/include

QMAKE_RPATHDIR += ../../clib

HEADERS +=$$files($$PWD/include/*.h)
SOURCES +=$$files($$PWD/src/*.cpp)
SOURCES +=$$files($$PWD/*.cpp)
