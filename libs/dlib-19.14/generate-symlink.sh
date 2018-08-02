#!/bin/bash
#
# Generate symbolic links for opencv library
#
dlibloc="/tmp/dlib"

if [ $# -eq 1 ]
  then
  dlibloc=$1
fi
cd ..
current=`pwd`

echo 'Dlib symlink selected: ' ${dlibloc}

sudo rm -rf ${dlibloc}
ln -s ${current}/dlib-19.14 ${dlibloc}
rm ${dlibloc}/dlib/all/module.mk
touch ${dlibloc}/dlib/all/module.mk
echo "DLIB_SOURCE=${dlibloc}/" > ${dlibloc}/dlib/all/module.mk
cat ${dlibloc}/dlib/all/module-trunc.mk >> ${dlibloc}/dlib/all/module.mk
