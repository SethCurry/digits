#set( leap_SOURCES Leap.h)
#add_library( leap )
#target_sources( leap PUBLIC ${libdigits_SOURCES})

#target_include_directories(leap INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# Import a prebuilt library to a CMake target
add_library(leap SHARED IMPORTED GLOBAL)
set_target_properties(leap PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/libLeap.so"
)

target_include_directories(leap INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
