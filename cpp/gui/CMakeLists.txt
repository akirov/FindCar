cmake_minimum_required(VERSION 3.14)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


# Set Qt location if needed
if (MINGW AND (CMAKE_SIZEOF_VOID_P EQUAL 4))
  #set(Qt5_DIR "D:/Qt/5.12.9/mingw73_32/lib/cmake/Qt5")
  set(Qt5_DIR "D:/Qt/5.15.2/mingw81_32/lib/cmake/Qt5")
elseif (MSVC AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
  #set(Qt5_DIR "D:/Qt/5.12.9/msvc2015_64/lib/cmake/Qt5")
  set(Qt5_DIR "D:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5")
elseif (LINUX)
  #> CMAKE_PREFIX_PATH=/opt/Qt/5.9.8/gcc_64/lib/cmake cmake ..
endif ()

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
message(STATUS "Found Qt${QT_VERSION_MAJOR}")


include_directories( ${LIBRARY_DIR} )


set(PROJECT_SOURCES
        main.cpp
        qtui.cpp
        qtui.h
        qtui.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(${GUI_EXE_NAME}
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET ${GUI_EXE_NAME} APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(${GUI_EXE_NAME} SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(${GUI_EXE_NAME}
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# Should we use PRIVATE?
target_link_libraries(${GUI_EXE_NAME} ${LIBRARY_NAME})
target_link_libraries(${GUI_EXE_NAME} Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(${GUI_EXE_NAME} ${OpenCV_LIBS})

set_target_properties(${GUI_EXE_NAME} PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${GUI_EXE_NAME})
endif()

install (TARGETS ${GUI_EXE_NAME} DESTINATION ${INSTALL_DIR})
