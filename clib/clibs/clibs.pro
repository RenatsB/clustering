include($${PWD}/../../common.pri)
TEMPLATE = lib
TARGET = clibs
DESTDIR = ../

OBJECTS_DIR = obj

INCLUDEPATH+=$$PWD/include

HEADERS += $$files($$PWD/include/*.hpp)
SOURCES += $$files($$PWD/src/*.cpp)
