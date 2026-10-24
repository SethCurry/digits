add_library(libdigitspy SHARED
    libdigitspy.cpp
    include/libdigitspy.hpp
)

target_compile_features(libdigitspy PUBLIC cxx_std_17)

target_include_directories(libdigitspy PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)