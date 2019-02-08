# This specifies the exe name
TARGET=NoiseSerial
# where to put the .o files
OBJECTS_DIR=obj
# core Qt Libs to use add more here if needed.
QT+=gui opengl core

# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
# where to put moc auto generated files
MOC_DIR=moc
# on a mac we don't create a .app bundle file ( for ease of multiplatform use)
CONFIG += console c++17
CONFIG-=app_bundle
# Auto include all .cpp files in the project src directory (can specifiy individually if required)
INCLUDEPATH += $$PWD/include

HEADERS += $$files($$PWD/include/*.h)
SOURCES += $$files($$PWD/src/*.cpp)

OTHER_FILES += $$files(../README.md)
# where our exe is going to live (root of project)
DESTDIR=./

QMAKE_CXXFLAGS += -std=c++17
QMAKE_CXXFLAGS += -O0
