/*
 * Copyright (C) 2013   Christopher C. Hulbert
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY CHRISTOPHER C. HULBERT ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CHRISTOPHER C. HULBERT OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <iostream>
#include "matio.h"

int
main (int argc, char *argv[]){
    const char *filename = argv[1];
    int err = EXIT_SUCCESS;
    mat_t *mat;
    matvar_t *ing_matvar, *cell, *field;
    size_t ing_index = 0;
    const char *data;

    mat = Mat_Open(filename, MAT_ACC_RDONLY);
    if (NULL == mat) {
        fprintf(stderr, "Error opening a.mat\n");
        return EXIT_FAILURE;
    }
    ing_matvar = Mat_VarReadInfo(mat, "path");
    if (NULL == ing_matvar) {
        fprintf(stderr, "Error reading 'ing' variable information\n");
        err = EXIT_FAILURE;
    } else if (MAT_C_CELL != ing_matvar->class_type) {
        fprintf(stderr, "Variable 'ing' is not a cell-arrayn\n");
        err = EXIT_FAILURE;
    } else {
        cell = Mat_VarGetCell(ing_matvar, ing_index);
        if (NULL == cell) {
            fprintf(stderr, "Error getting 'ing{%lu}'\n", ing_index);
            err = EXIT_FAILURE;
        } else {
            auto celltype = cell->class_type;
            std::cout << "Cell type = " << celltype << std::endl;
            if (MAT_C_CHAR != cell->class_type) {
                fprintf(stderr, "Variable 'ing{%lu}' is not a struct-arrayn\n", ing_index);
                err = EXIT_FAILURE;
            } else {
                auto datatype = cell->data_type;
                auto size = cell->data_size;
                auto dim = cell->dims;

                //data = (const char *) cell->data;
                std::cout << datatype <<" with dim of " << dim << " and a size of " << size << std::endl;
                /*field = Mat_VarGetStructFieldByName(cell, ing_fieldname, 0);
                if (NULL == cell) {
                    fprintf(stderr, "Error getting 'ing{%lu}.%s'\n", ing_index, ing_fieldname);
                    err = EXIT_FAILURE;
                } else {
                    int read_err = Mat_VarReadDataAll(mat, field);
                    if (read_err) {
                        fprintf(stderr, "Error reading data for 'ing{%lu}.%s'\n", ing_index, ing_fieldname);
                        err = EXIT_FAILURE;
                    } else {
                        Mat_VarPrint(field, 1);
                    }
                }
            }
            Mat_VarFree(ing_matvar);
        }
        Mat_Close(mat);

        return err;*/
            }
        }
    }
}
//
// Created by larnal on 02/08/18.
//

