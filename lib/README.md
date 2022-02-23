Tệp tin bao gồm:
1. Tensorflow C Linux version 2.6.0
2. TensorflowLite Linux version 2.4.1 
3. Link tải về thư viện [link to library!](...)



bazel build -c opt //tensorflow/lite:libtensorflowlite.so
bazel build -c opt //tensorflow/lite/experimental/c:libtensorflowlite_c.so
