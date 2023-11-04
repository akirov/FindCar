cmake_minimum_required(VERSION 3.14)

include_directories( ${LIBRARY_DIR} )

add_executable(${CLI_EXE_NAME} main.cpp)

target_link_libraries(${CLI_EXE_NAME} ${LIBRARY_NAME} ${OpenCV_LIBS} ${Tesseract_LIBS} ${Leptonica_LIBS})

install(TARGETS ${CLI_EXE_NAME} DESTINATION ${INSTALL_DIR})
