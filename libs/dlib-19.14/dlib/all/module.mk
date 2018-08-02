DLIB_SOURCE=/tmp/dlib/
# add source files
SOURCE_PROJECT += source.cpp

# add header files
HEADER_PROJECT += -I$(DLIB_SOURCE)

SOURCE_PATH_PROJECT += $(DLIB_SOURCE)/dlib/all
