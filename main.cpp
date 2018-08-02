// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the support vector machine
    utilities from the dlib C++ Library.  

    This example creates a simple set of data to train on and then shows
    you how to use the cross validation and svm training functions
    to find a good decision function that can classify examples in our
    data set.


    The data used in this example will be 2 dimensional data and will
    come from a distribution where points with a distance less than 10
    from the origin are labeled +1 and all other points are labeled
    as -1.
        
*/


#include <iostream>
#include <dlib/svm.h>
#include <unordered_map>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>


using namespace std;
using namespace dlib;

template<template<int, template<typename> class, int, typename> class block, int N,
    template<typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template<template<int, template<typename> class, int, typename> class block, int N,
    template<typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,
                                                     2,
                                                     2,
                                                     2,
                                                     dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template<int N, template<typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template<int N, typename SUBNET> using ares      = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template<int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template<typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template<typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template<typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template<typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template<typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
    alevel0<
        alevel1<
            alevel2<
                alevel3<
                    alevel4<
                        dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
                                                                                     dlib::input_rgb_image_sized<150>
                        >>>>>>>>>>>>;

unordered_map<string, double> getCSV(const string &file, char delimiter){
    unordered_map<string, double> data {};
    string line, value1, value2;
    ifstream f(file, ios::in);
    if (!f) {
        cerr << "Failed to open file !";
        exit(1);
    }
    auto nline = 0;
    cout << "reading CSV file " << file << endl;

    while (getline(f, line)) {
        getline(std::stringstream(line), value1, delimiter);
        getline(std::stringstream(line), value2);
        data[value1] = stod(value2);
        nline++;
    }
    cout << "Opened CSV file with " << nline << " lines." << endl;
    return data;
}

std::vector<matrix<rgb_pixel>> getFaceShape(const unordered_map<string,
                                                                double> &map, const std::string &pathShaper, int max = 0){

    dlib::shape_predictor shaper;

    deserialize(pathShaper) >> shaper;

    auto shapes = std::vector<matrix<rgb_pixel>> {};
    auto detector = dlib::get_frontal_face_detector();
    auto shape = full_object_detection{};
    auto face_chip = matrix<rgb_pixel> {};
    auto img = matrix<rgb_pixel>{};

    auto it = map.begin();
    auto i = 0;
    if (max == 0) max = (int) map.size();
    while (it != map.end() && i < max) {
        cout << "analysing image " << i << "..." << endl;
        auto path = "/home/larnal/workspace/CLion/HeaseRobotics/svm_trainer/data/wiki_crop/" + it->first;
        load_image(img, path);
        auto rects = detector(img);
/*        if (rects.empty()) {
            cout << "Image has no faces" << endl;
        } else if (rects.size() > 1) {
            cout << "Image has multiple faces" << endl;
        } else {*/
        if (rects.size() == 1) {
            shape = shaper(img, rects[0]);
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            shapes.emplace_back(face_chip);
        }
        it++;
        i++;
    }
    return shapes;
}

std::vector<float> fillLabels(const unordered_map<string, double> &map, int max){
    std::vector<float> values;
    auto it = map.begin();
    auto i = 0;
    while (it != map.end() && i < max) {
        values.emplace_back((float) it->second);
        it++;
        i++;
    }
    return values;
}

void checkLabels(std::unordered_map<string, double> &map) {
    // reformat data : remove unidentified gender, change female label to -1
    auto initsize = map.size();
    auto it = map.begin();
    while (it != map.end()) {
        if (it->second == 0) it->second = -1.;
        if (it->second != -1 && it->second != 1) it = map.erase(it);
        else it++;
    }
    cout << initsize - map.size() << "bad labels removed." << endl;
}



int main(){
    auto nsamples = 10000;
    auto pathCSV = "/home/jerome/workspace/Gender_Detection/svm_trainer/data//wiki_crop/data.csv";
    auto CSVdelimiter = ',';
    auto pathShaper = "/home/jerome/workspace/Gender_Detection/svm_trainer/data/models/shape_predictor_5_face_landmarks.dat";
    auto pathDescripter = "/home/jerome/workspace/Gender_Detection/svm_trainer/data/models/dlib_face_recognition_resnet_model_v1.dat";
    auto outputSVM = "gender_recognizer.dat";
    // The svm functions use column vectors to contain a lot of the data on which they
    // operate. So the first thing we do here is declare a convenient typedef.  

    // This typedef declares a matrix with 2 rows and 1 column.  It will be the object that
    // contains each of our 2 dimensional samples.   (Note that if you wanted more than 2
    // features in this vector you can simply change the 2 to something else.  Or if you
    // don't know how many features you want until runtime then you can put a 0 here and
    // use the matrix.set_size() member function)
    typedef matrix<float, 0, 1> sample_type; //128D vector

    // This is a typedef for the type of kernel we are going to use in this example.  In
    // this case I have selected the radial basis kernel that can operate on our 2D
    // sample_type objects
    typedef radial_basis_kernel<sample_type> kernel_type;


    // Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<float> labels;

    // Now let's put some data into our samples and labels objects.  We do this by looping
    // over a bunch of points and labeling them according to their distance from the
    // origin.

    cout << "getting samples data..." << endl;
    auto train_data = getCSV(pathCSV, CSVdelimiter);
    checkLabels(train_data);

    auto shapes = getFaceShape(train_data, pathShaper, nsamples);
    cout << shapes.size() << " faces identified." << endl;
    anet_type descripter;
    deserialize(pathDescripter) >> descripter;
    samples = descripter(shapes);
    cout << samples.size() << " 128Dvectors created from faces." << endl;
    labels = fillLabels(train_data, nsamples);
    cout << samples.size() << " labels extracted." << endl;

    cout << "trying normalization..." << endl;
    vector_normalizer<sample_type> normalizer;
    // Let the normalizer learn the mean and standard deviation of the samples.
    normalizer.train(samples);
    // now normalize each sample
    cout << "Changing samples for normalized samples..." << endl;
    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);

    // Now that we have some data we want to train on it.  However, there are two
    // parameters to the training.  These are the nu and gamma parameters.  Our choice for
    // these parameters will influence how good the resulting decision function is.  To
    // test how good a particular choice of these parameters is we can use the
    // cross_validate_trainer() function to perform n-fold cross validation on our training
    // data.  However, there is a problem with the way we have sampled our distribution
    // above.  The problem is that there is a definite ordering to the samples.  That is,
    // the first half of the samples look like they are from a different distribution than
    // the second half.  This would screw up the cross validation process but we can fix it
    // by randomizing the order of the samples with the following function call.
    cout << "randomizing samples..." << endl;
    randomize_samples(samples, labels);


    // The nu parameter has a maximum value that is dependent on the ratio of the +1 to -1
    // labels in the training data.  This function finds that value.
    // here we make an instance of the svm_nu_trainer object that uses our kernel type.
    svm_c_trainer<kernel_type> trainer;

    cout << "doing cross validation" << endl;
    for (double C = 1; C < 1000000; C *= 5) {
    for (double gamma = 0.00625; gamma <= 10; gamma *= 5) {

            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(gamma));
            trainer.set_c(C);

            cout << "gamma: " << gamma << "    C: " << C;
            // Print out the cross validation accuracy for 3-fold cross validation using
            // the current gamma and C.  cross_validate_trainer() returns a row vector.
            // The first element of the vector is the fraction of +1 training examples
            // correctly classified and the second number is the fraction of -1 training
            // examples correctly classified.
            cout << "     cross validation accuracy: "
                 << cross_validate_trainer(trainer, samples, labels, 3);
        }
    }


/*
// From looking at the output of the above loop it turns out that good
// values for C and gamma for this problem are 5 and 0.15625 respectively.
// So that is what we will use.

// Now we train on the full set of data and obtain the resulting decision
// function.  The decision function will return values >= 0 for samples it
// predicts are in the +1 class and numbers < 0 for samples it predicts to
// be in the -1 class.
    trainer.set_kernel(kernel_type(0.15625));
    trainer.set_c(10);
    typedef decision_function<kernel_type> dec_funct_type;

// Here we are making an instance of the normalized_function object.  This object
// provides a convenient way to store the vector normalization information along with
// the decision function we are going to learn.
    dec_funct_type learned_function = trainer.train(samples, labels); // perform the actual SVM training and save the results


    serialize(outputSVM) << learned_function;*/
}






