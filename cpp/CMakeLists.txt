cmake_minimum_required(VERSION 3.14)

project(FindCar VERSION 0.1 LANGUAGES CXX)

set(LIBRARY_NAME imgtotxt)
set(CLI_EXE_NAME ${PROJECT_NAME}CLI)
set(GUI_EXE_NAME ${PROJECT_NAME}GUI)


# Set OpenCV location, if needed
if (MINGW AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  #set(OpenCV_DIR "D:/OpenCV/3.4.5/mingw730_32/install")
  set(OpenCV_DIR "D:/OpenCV/3.4.16/mingw810_32/install")
elseif (MSVC AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
  #set(OpenCV_DIR "D:/OpenCV/3.4.5/vs2015/install/x64/vc14/lib")
  set(OpenCV_DIR "D:/OpenCV/4.6.0/msvc2019_64/install/x64/vc16/lib")
endif ()

find_package(OpenCV REQUIRED)


# Set Tesseract and Leptonica locations
if (MINGW AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  #set(Tesseract_DIR ${CMAKE_SOURCE_DIR}/../tesseract/bin/mingw_win32/lib/cmake/tesseract)
  set(TESSERACT_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../tesseract/include")
  set(TESSERACT_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/../tesseract/bin/mingw_win32/lib")
  set(Tesseract_LIBS tesseract53)
  set(LEPTONICA_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../tesseract/include")
  set(LEPTONICA_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/../tesseract/bin/mingw_win32/lib")
  set(Leptonica_LIBS leptonica-1.83.1)
else()
  message(STATUS "Please set Tesseract and Leptonica locations")
endif()

#find_package(Tesseract REQUIRED)
include_directories( ${TESSERACT_INCLUDE_DIRS} )
include_directories( ${LEPTONICA_INCLUDE_DIRS} )
link_directories( ${TESSERACT_LIBRARY_DIRS} )
link_directories( ${LEPTONICA_LIBRARY_DIRS} )


# Use C++ standard (98, 11, 14, 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX_EXTENSIONS OFF)


# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)


#add_definitions(-D${SOMEVAR})


if (MINGW)
  set(TOOL_TRG "mingw_win")
elseif (MSVC)
  set(TOOL_TRG "msvc_win")
elseif (LINUX)
  #TODO be more specific
  set(TOOL_TRG "linux")
endif ()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TRG_PLAT "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(TRG_PLAT "32")
endif()

set(INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin/${TOOL_TRG}${TRG_PLAT})
message(STATUS "Install folder: ${INSTALL_DIR}")

set(LIBRARY_DIR ${CMAKE_SOURCE_DIR}/lib)


add_subdirectory(lib)
add_subdirectory(cli)
add_subdirectory(gui)
