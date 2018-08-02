
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

std::vector<matrix<rgb_pixel>> getFaceShape(
        const unordered_map<string,
                double> &map, const std::string &pathShaper, const std::string &dataPath, int max = 0) {

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
        auto path = dataPath + it->first;
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
    cout << initsize - map.size() << " bad labels removed." << endl;
}



int main(){
    auto nsamples = 10000;
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

    auto shapes = getFaceShape(train_data, pathShaper, projectPath + "data/wiki_crop/", nsamples);
    cout << shapes.size() << " faces identified." << endl;

    anet_type descripter;
    deserialize(pathDescripter) >> descripter;
    samples = descripter(shapes);

    cout << samples.size() << " samples created from shapes." << endl;
    labels = fillLabels(train_data, nsamples);
    cout << samples.size() << " labels extracted." << endl;

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

    // Performing cross validation to choose gamma & C parameter values
    cout << "doing cross validation" << endl;
    for (double C = 1; C < 1000000; C *= 5) {
    for (double gamma = 0.00625; gamma <= 10; gamma *= 5) {

            // tell the trainer the parameters we want to use
            trainer.set_kernel(kernel_type(static_cast<const float>(gamma)));
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

    //Stop there to choose the right gamma & C parameters
    exit(0);


    // Now we train on the full set of data and obtain the resulting decision
    // function.  The decision function will return values >= 0 for samples it
    // predicts are in the +1 class and numbers < 0 for samples it predicts to
    // be in the -1 class.
    trainer.set_kernel(kernel_type(0.15625));
    trainer.set_c(10);
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;


    // Object  to store the descision function alongside normalized samples
    funct_type learned_function;
    learned_function.normalizer = normalizer;
    // perform the actual SVM training and save the results
    learned_function.function = trainer.train(samples, labels);

    // save the SVM in a file
    serialize(outputSVM) << learned_function;
}






