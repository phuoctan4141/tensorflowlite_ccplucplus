Tham khảo code tại [link to YijinLiu!](https://github.com/YijinLiu/tf-cpu)

### TensorData:
match the datas that the model expects
```
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
```

### FeedInMat:
convert Mat to the same input data type
```
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
```

### AnnotateMat:
find face class in the image and draw bbox
```
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
```

# Run the code:
`gcc -I/usr/local/include -I/usr/include/opencv4 -L/lib img_detection.cpp -o test -lstdc++ -ltensorflowlite -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_highgui`
