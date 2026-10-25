add_library(libdigits SHARED
    listener.cpp
    logging.cpp
)

target_compile_features(libdigits PUBLIC cxx_std_17)

target_include_directories(libdigits PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/include"
)
