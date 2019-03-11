TEMPLATE = subdirs
SUBDIRS = serial/clusteringS.pro parallel/clusteringP.pro

OTHER_FILES += $$files($$PWD/README.md)
