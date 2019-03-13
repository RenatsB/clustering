#DEPPATH = $${PWD}/dep
#DEPS = $$system(ls $${DEPPATH})
#!isEmpty(DEPS) {
#  for(d, DEPS) {
#    INCLUDEPATH += $${DEPPATH}/$${d}
#    INCLUDEPATH += $${DEPPATH}/$${d}/include
#  }
#}

isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}

INCLUDEPATH += $${PWD}/generator/include

#Linker search paths
#LIBS += -L/home/s4902673/SuiteSparse/lib
# Linker libraries
#LIBS += -lcholmod

#DEFINES += FLO_USE_DOUBLE_PRECISION
#DEFINES += THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

QT -= opengl core gui
CONFIG += console c++14
CONFIG -= app_bundle

# Standard flags
QMAKE_CXXFLAGS += -std=c++14 -g -fdiagnostics-color
# Optimisation flags
QMAKE_CXXFLAGS += -Ofast -march=native -frename-registers -funroll-loops -ffast-math -fassociative-math
# Intrinsics flags
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -msse -msse2 -msse3
# Enable all warnings
QMAKE_CXXFLAGS += -Wall -Wextra -pedantic-errors -Wno-sign-compare
