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
