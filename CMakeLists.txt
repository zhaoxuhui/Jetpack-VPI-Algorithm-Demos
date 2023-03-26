cmake_minimum_required(VERSION 3.13)
project(vpi_algorithm_examples)

set(CMAKE_CXX_STANDARD 11)

find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(demo_boxFilter cpp/demo_boxFilter.cpp)
target_link_libraries(demo_boxFilter vpi ${OpenCV_LIBS})

add_executable(demo_bilateralFilter cpp/demo_bilateralFilter.cpp)
target_link_libraries(demo_bilateralFilter vpi ${OpenCV_LIBS})

add_executable(demo_gaussianFilter cpp/demo_gaussianFilter.cpp)
target_link_libraries(demo_gaussianFilter vpi ${OpenCV_LIBS})

add_executable(demo_gaussianPyramid cpp/demo_gaussianPyramid.cpp)
target_link_libraries(demo_gaussianPyramid vpi ${OpenCV_LIBS})

add_executable(demo_laplacianPyramid cpp/demo_laplacianPyramid.cpp)
target_link_libraries(demo_laplacianPyramid vpi ${OpenCV_LIBS})

add_executable(demo_erode cpp/demo_erode.cpp)
target_link_libraries(demo_erode vpi ${OpenCV_LIBS})

add_executable(demo_dilate cpp/demo_dilate.cpp)
target_link_libraries(demo_dilate vpi ${OpenCV_LIBS})

add_executable(demo_convolution cpp/demo_convolution.cpp)
target_link_libraries(demo_convolution vpi ${OpenCV_LIBS})

add_executable(demo_sepConvolution cpp/demo_sepConvolution.cpp)
target_link_libraries(demo_sepConvolution vpi ${OpenCV_LIBS})

add_executable(demo_cvtFormat cpp/demo_cvtFormat.cpp)
target_link_libraries(demo_cvtFormat vpi ${OpenCV_LIBS})

add_executable(demo_rescale cpp/demo_rescale.cpp)
target_link_libraries(demo_rescale vpi ${OpenCV_LIBS})

add_executable(demo_remap cpp/demo_remap.cpp)
target_link_libraries(demo_remap vpi ${OpenCV_LIBS})

add_executable(demo_fft cpp/demo_fft.cpp)
target_link_libraries(demo_fft vpi ${OpenCV_LIBS})

add_executable(demo_ifft cpp/demo_ifft.cpp)
target_link_libraries(demo_ifft vpi ${OpenCV_LIBS})

add_executable(demo_lensDistCorrect cpp/demo_lensDistCorrect.cpp)
target_link_libraries(demo_lensDistCorrect vpi ${OpenCV_LIBS})

add_executable(demo_stereoDisparity cpp/demo_stereoDisparity.cpp)
target_link_libraries(demo_stereoDisparity vpi ${OpenCV_LIBS})

add_executable(demo_harrisDetector cpp/demo_harrisDetector.cpp)
target_link_libraries(demo_harrisDetector vpi ${OpenCV_LIBS})

add_executable(demo_histogram cpp/demo_histogram.cpp)
target_link_libraries(demo_histogram vpi ${OpenCV_LIBS})

add_executable(demo_equalizeHist cpp/demo_equalizeHist.cpp)
target_link_libraries(demo_equalizeHist vpi ${OpenCV_LIBS})

add_executable(demo_minMaxLoc cpp/demo_minMaxLoc.cpp)
target_link_libraries(demo_minMaxLoc vpi ${OpenCV_LIBS})

add_executable(demo_imgFlip cpp/demo_imgFlip.cpp)
target_link_libraries(demo_imgFlip vpi ${OpenCV_LIBS})

add_executable(demo_medianFilter cpp/demo_medianFilter.cpp)
target_link_libraries(demo_medianFilter vpi ${OpenCV_LIBS})

add_executable(demo_FASTDetector cpp/demo_FASTDetector.cpp)
target_link_libraries(demo_FASTDetector vpi ${OpenCV_LIBS})

add_executable(demo_mixChannels cpp/demo_mixChannels.cpp)
target_link_libraries(demo_mixChannels vpi ${OpenCV_LIBS})

add_executable(demo_cannyEdge cpp/demo_cannyEdge.cpp)
target_link_libraries(demo_cannyEdge vpi ${OpenCV_LIBS})

add_executable(demo_ORBDetector cpp/demo_ORBDetector.cpp)
target_link_libraries(demo_ORBDetector vpi ${OpenCV_LIBS})

add_executable(demo_imgStatistics cpp/demo_imgStatistics.cpp)
target_link_libraries(demo_imgStatistics vpi ${OpenCV_LIBS})

add_executable(demo_templateMatching cpp/demo_templateMatching.cpp)
target_link_libraries(demo_templateMatching vpi ${OpenCV_LIBS})