cmake_minimum_required(VERSION 3.14)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(LIBRARY_SOURCES
  OcvTextDetector.hpp
  OcvTextDetector.cpp
  TessTextReader.hpp
  TessTextReader.cpp
)

#add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SOURCES})
add_library(${LIBRARY_NAME} STATIC ${LIBRARY_SOURCES})

#install(TARGETS ${LIBRARY_NAME} DESTINATION ${INSTALL_DIR})
