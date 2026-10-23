
set( libdigits_SOURCES cli.cpp logging.cpp listener.cpp )
add_library( libdigits SHARED )
target_sources( libdigits PUBLIC ${libdigits_SOURCES})

target_include_directories(libdigits INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
