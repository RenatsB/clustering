TEMPLATE = subdirs
SUBDIRS = clib demo #test

demo.depends = clib
#test.depends = clib

OTHER_FILES += $$files($$PWD/README.md)
