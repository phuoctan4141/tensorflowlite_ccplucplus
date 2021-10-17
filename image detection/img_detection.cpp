#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp> // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp> // OpenCV window I/O

#include <algorithm>
#include <queue>
#include <tuple>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <math.h>
#include <stdio.h>
#include <cstdlib>
#include <time.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace cv;
using namespace std;
using namespace tflite;

#define IMAGE_MEAN 128.0f
#define IMAGE_STD 128.0f
#define resize_with 320
#define resize_height 320
#define channels 3

template <typename T>
T *TensorData(TfLiteTensor *tensor);

template <>
float *TensorData(TfLiteTensor *tensor)
{
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++)
        nelems *= tensor->dims->data[i];
    switch (tensor->type)
    {
    case kTfLiteFloat32:
        return tensor->data.f + nelems / (resize_with * resize_height * channels);
    default:
        cerr << "Should not reach here!" << endl;
    }
    return nullptr;
}

template <>
uint8_t *TensorData(TfLiteTensor *tensor)
{
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++)
        nelems *= tensor->dims->data[i];
    switch (tensor->type)
    {
    case kTfLiteUInt8:
        return tensor->data.uint8 + nelems / (resize_with * resize_height * channels);
    default:
        cerr << "Should not reach here!" << endl;
    }
    return nullptr;
}

unique_ptr<FlatBufferModel> model;
unique_ptr<Interpreter> uni_interpreter;

TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_locations = nullptr;
TfLiteTensor *output_classes = nullptr;
TfLiteTensor *output_scores = nullptr;
TfLiteTensor *num_detections = nullptr;

clock_t start_clock, end_clock;

void draw_box(Mat &rect_image, int xmin, int ymin, int xmax, int ymax)
{
    rectangle(rect_image, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 2);
}

void FeedInMat(const Mat &mat)
{
    switch (input_tensor->type)
    {
    case kTfLiteFloat32:
    {
        float *dst = TensorData<float>(input_tensor);
        const int row_elems = (resize_with * channels);
        for (int row = 0; row < (resize_height * channels); row++)
        {
            const uchar *row_ptr = mat.ptr(row);
            for (int i = 0; i < row_elems; i++)
            {
                dst[i] = (row_ptr[i] - IMAGE_MEAN) / IMAGE_STD;
            }
            dst += row_elems;
        }
    }
    break;
    case kTfLiteUInt8:
    {
        uint8_t *dst = TensorData<uint8_t>(input_tensor);
        const int row_elems = (resize_with * channels);
        for (int row = 0; row < (resize_height * channels); row++)
        {
            memcpy(dst, mat.ptr(row), row_elems);
            dst += row_elems;
        }
    }
    break;
    default:
        cerr << "Should not reach here!" << endl;
    }
}

void AnnotateMat(Mat &mat)
{
    const float *detection_locations_ = TensorData<float>(output_locations);
    const float *detection_classes_ = TensorData<float>(output_classes);
    const float *detection_scores_ = TensorData<float>(output_scores);
    const int num_detections_ = *TensorData<float>(num_detections);

    for (int d = 0; d < num_detections_; d++)
    {
        start_clock = clock();
        const float score = detection_scores_[d];
        const int ymin = detection_locations_[4 * d] * mat.rows;
        const int xmin = detection_locations_[4 * d + 1] * mat.cols;
        const int ymax = detection_locations_[4 * d + 2] * mat.rows;
        const int xmax = detection_locations_[4 * d + 3] * mat.cols;
        if (score < .3f)
        {
            cout << "Ignore detection " << d << " of '"
                 << "' with score " << score
                 << " @[" << xmin << "," << ymin << ":" << xmax << "," << ymax << "]" << endl;
        }
        else
        {
            cout << "Detected " << d << " of '"
                 << "' with score " << score
                 << " @[" << xmin << "," << ymin << ":" << xmax << "," << ymax << "]" << endl;
            draw_box(mat, xmin, ymin, xmax, ymax);
        }
        cout << "Time to run: " << setprecision(10) << (double)(clock() - start_clock) / CLOCKS_PER_SEC << endl;
    }
}

int main(int argc, char **argv)
{
    end_clock = clock();
    // Read the image file
    Mat image = imread("damtv.jpg", IMREAD_COLOR);

    // Check for failure
    if (image.empty())
    {
        cerr << "Could not open or find the image" << endl;
        return -1;
    }

    Mat resized_image;
    resize(image, resized_image, Size(resize_with, resize_with), INTER_LINEAR);

    // Create model from file. Note that the model instance must outlive the
    // interpreter instance.
    auto model = FlatBufferModel::BuildFromFile("Face_Detection.tflite");
    if (model == nullptr)
    {
        cerr << "Model not found!" << endl;
        return -1;
    }
    // Create an Interpreter with an InterpreterBuilder.
    ops::builtin::BuiltinOpResolver resolver;
    if (InterpreterBuilder(*model, resolver)(&uni_interpreter) != kTfLiteOk)
    {
        cerr << "Failed to build interpreter!" << endl;
        return -1;
    }
    if (uni_interpreter->AllocateTensors() != kTfLiteOk)
    {
        cerr << "Failed to build interpreter!" << endl;
        return -1;
    }

    input_tensor = uni_interpreter->tensor(uni_interpreter->inputs()[0]);

    output_locations = uni_interpreter->tensor(uni_interpreter->outputs()[0]);
    output_classes = uni_interpreter->tensor(uni_interpreter->outputs()[1]);
    output_scores = uni_interpreter->tensor(uni_interpreter->outputs()[2]);
    num_detections = uni_interpreter->tensor(uni_interpreter->outputs()[3]);

    Mat tf_image = resized_image;
    FeedInMat(tf_image);

    uni_interpreter->Invoke();
    AnnotateMat(tf_image);

    cout << "Full time: " << setprecision(10) << (double)(clock() - end_clock) / CLOCKS_PER_SEC << endl;

    //Show tf_image in the window
    String windowName = "The Face"; //Name of the window

    namedWindow(windowName); // Create a window

    imshow(windowName, tf_image); // Show our image inside the created window.

    waitKey(0); // Wait for any keystroke in the window

    destroyWindow(windowName); //destroy the created window

    return 0;
}
