#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "matio.h"


int main(int argc,char **argv)
{
    mat_t    *matfp;
    matvar_t *matvar;

    matfp = Mat_Open(argv[1],MAT_ACC_RDONLY);
    if ( NULL == matfp ) {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n",argv[1]);
        return EXIT_FAILURE;
    }

    while ( (matvar = Mat_VarReadNext(matfp)) != NULL ) {
        auto nametest = std::string(matvar->name);
        std::cout << nametest << std::endl;
        if (nametest == "gd") {
            int type = matvar->data_type;
            std::cout << "type = " << type << std::endl;
            auto rank = matvar->rank;
            std::cout << "rank = " << rank << std::endl;

            auto dims = matvar->dims;
            auto size = dims[1];
            std::cout << "size = " << size << std::endl;
            std::vector<double> vec;
            void *data = matvar->data;
            if (data != NULL) {
                vec.reserve(size);
                vec.assign((double*) data, ((double*)data ) + size);
            }
            for (auto &i : vec) {
                std::cout << "value = "<< i << std::endl;
            }
        }

        if (nametest == "path") {
            int type = matvar->data_type;
            std::cout << "type = " << type << std::endl;
            auto rank = matvar->rank;
            std::cout << "rank = " << rank << std::endl;

            auto dims = matvar->dims;
            auto size = dims[1];
            std::cout << "size = " << size << std::endl;



            //auto number = *( (double*) matvar->data);

        }
        Mat_VarFree(matvar);
        matvar = NULL;
    }

    Mat_Close(matfp);
    return EXIT_SUCCESS;
}