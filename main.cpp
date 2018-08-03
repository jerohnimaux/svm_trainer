
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <dlib/svm.h>
#include <unordered_map>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>


using namespace std;
using namespace dlib;

//Tyedefs for the descripter object
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
//----------------------------------------------------------------------------------------------------------------------

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


unordered_map<string, double> getCSV(const string &file, char delimiter) {
    unordered_map<string, double> data{};

    string line, value1, value2;
    ifstream f(file, ios::in);
    if (!f) {
        cerr << "Failed to open file !";
        exit(1);
    }
    cout << "reading CSV file " << file << endl;

    while (getline(f, line)) {
        auto elements = split(line, delimiter);
        if (elements.size() != 2) {
            cout << "error with CSV parsing." << endl;
        }
        data[elements[0]] = stod(elements[1]);
    }
    cout << "Opened CSV file with " << data.size() << " lines." << endl;
    return data;
}


void checkLabels(std::unordered_map<string, double> &map) {
    // reformat data : remove unidentified gender, change female label to -1
    auto initsize = map.size();
    auto it = map.begin();
    while (it != map.end()) {
        if (it->second < 0.1) it->second = -1.;
        if (isnan(it->second)) it = map.erase(it);
        else it++;
    }
    cout << initsize - map.size() << " bad labels removed." << endl;
}


std::vector<matrix<rgb_pixel>> getFaceShape(
        const unordered_map<string,
                double> &map,
        const std::string &pathShaper,
        const std::string &dataPath,
        std::vector<float> &labels,
        int max = 0) {

    dlib::shape_predictor shaper;

    deserialize(pathShaper) >> shaper;

    auto shapes = std::vector<matrix<rgb_pixel>>{};
    auto detector = dlib::get_frontal_face_detector();
    auto shape = full_object_detection{};
    auto face_chip = matrix<rgb_pixel>{};
    auto img = matrix<rgb_pixel>{};

    auto it = map.begin();
    auto i = 0;
    if (max == 0) max = (int) map.size();
    while (it != map.end() && i < max) {
        cout << "analysing image " << i << " of " << max << "...  ";
        auto path = dataPath + it->first;
        load_image(img, path);
        auto rects = detector(img);
        if (rects.size() == 1) {
            shape = shaper(img, rects[0]);
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            shapes.emplace_back(face_chip);
            labels.emplace_back(it->second);
            cout << "image validated.";
        }
        cout << endl;
        it++;
        i++;
    }
    return shapes;
}

template<typename T, typename S>
std::tuple<float, float> cross_validation(
        svm_c_trainer<T> trainer,
        const std::vector<S> &samples,
        const std::vector<float> &labels,
        float Cmin,
        float Cmax,
        float gammaMin,
        float gammaMax,
        int step = 5,
        int times = 3) {
    // Performing cross validation to choose gamma & C parameter values
    float gammaPower = std::pow(gammaMax / gammaMin, 1.0f / step);
    float Cpower = std::pow(Cmax / Cmin, 1.0f / step);
    cout << "doing cross validation" << endl;
    std::vector<std::tuple<float, float, float, float>> parameters;
    for (float gamma = gammaMin; gamma < gammaMax; gamma *= gammaPower) {
        for (float C = Cmin; C < Cmax; C *= Cpower) {
            // tell the trainer the parameters we want to use
            trainer.set_kernel(T(gamma));
            trainer.set_c(C);

            std::ostringstream str;
            str << cross_validate_trainer(trainer, samples, labels, times);
            std::istringstream iss(str.str());
            std::vector<std::string> result{
                    std::istream_iterator<std::string>(iss), {}
            };
            if (result.size() != 2)
                throw std::logic_error("error during cross validation. Returned values are different of 2.");
            parameters.emplace_back(std::make_tuple(C, gamma, stof(result[0]), stof(result[1])));
        }
    }
    auto best_p = parameters[0];
    for (auto &p : parameters) {
        if (std::get<2>(p) + std::get<3>(p) > std::get<2>(best_p) + std::get<3>(best_p)) {
            best_p = p;
        }
    }
    return std::make_tuple(std::get<0>(best_p), std::get<1>(best_p));
}


int main() {
    auto nsamples = 2000;
    auto projectPath = std::string("/home/jerome/workspace/Gender_Detection/svm_trainer/");
    auto csv_file = "data/wiki_crop/data.csv";
    auto CSVdelimiter = ',';
    auto shaper_file = "data/models/shape_predictor_5_face_landmarks.dat";
    auto descripter_file = "data/models/dlib_face_recognition_resnet_model_v1.dat";
    auto output_file = "gender_recognizer.dat";

    auto pathCSV = projectPath + csv_file;
    auto pathShaper = projectPath + shaper_file;
    auto pathDescripter = projectPath + descripter_file;
    auto outputSVM = projectPath + output_file;


    // type of the sample
    typedef matrix<float, 0, 1> sample_type; //128D vector

    // type of kernel for the SVM. Here we choose rbf kernel
    typedef radial_basis_kernel<sample_type> kernel_type;

    // samples and labels vectors used to train the SVM
    std::vector<sample_type> samples;
    std::vector<float> labels;

    // Getting the Data
    cout << "getting samples data..." << endl;
    auto train_data = getCSV(pathCSV, CSVdelimiter);
    checkLabels(train_data);

    // Fill samples and labels vector with 128D vectors and labels
    auto shapes = getFaceShape(train_data, pathShaper, projectPath + "data/wiki_crop/", labels, nsamples);
    cout << shapes.size() << " faces identified." << endl;

    anet_type descripter;
    deserialize(pathDescripter) >> descripter;
    samples = descripter(shapes);

    cout << samples.size() << " samples created from shapes." << endl;
    cout << labels.size() << " labels extracted." << endl;

    // Normalizing data
    cout << "trying normalization..." << endl;
    vector_normalizer<sample_type> normalizer;
    // Let the normalizer learn the mean and standard deviation of the samples.
    normalizer.train(samples);
    // now normalize each sample
    cout << "Changing samples for normalized samples..." << endl;
    for (auto &sample : samples)
        sample = normalizer(sample);

    // Randomizing distribution of samples in the vector
    cout << "randomizing samples..." << endl;
    randomize_samples(samples, labels);

    // Creating the SVM
    svm_c_trainer<kernel_type> trainer;
    /*float C, gamma;
    try {
        std::tie(C, gamma) = cross_validation(trainer, samples, labels, 1, 5, 0.0001, 0.1);
    } catch (std::logic_error &e){
        cout << e.what() << endl;
    }*/


    // Now we train on the full set of data and obtain the resulting decision
    // function.  The decision function will return values >= 0 for samples it
    // predicts are in the +1 class and numbers < 0 for samples it predicts to
    // be in the -1 class.
    trainer.set_kernel(kernel_type(0.00238247));
    trainer.set_c(5);
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;


    // Object  to store the descision function alongside normalized samples
    funct_type learned_function;
    learned_function.normalizer = normalizer;
    // perform the actual SVM training and save the results
    cout << "Start the training...  ";
    learned_function.function = trainer.train(samples, labels);
    cout << "Training finished !" << endl;

    // save the SVM in a file
    cout << "Saving the SVM to " << outputSVM << endl;
    serialize(outputSVM) << learned_function;
    cout << "save successful !" << endl;
}






