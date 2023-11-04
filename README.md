FindCar
=======

Finds a car with given number plate in pictures or videos. Builds with C++. Uses Tesseract[^1], OpenCV[^2] and Qt[^3] libraries.

Tesseract v5.3.1 win32 binaries, entirely built from sources with mingw810_32 g++ from Qt 5.15.2 SDK on Windows 10, together with header files, is included in the project.
If you want to use it, add FindCar/tesseract/bin/mingw_win32/bin to PATH, and set TESSDATA_PREFIX environment variable to tesseract/tessdata folder.


## FindCar CLI and GUI

Required C++ libraries: Tesseract (if not using included mingw32 build), OpenCV, Qt (for the GUI).
You may want to adjust OpenCV, Qt and probably Tesseract locations in CMakeLists.txt files.
Build with a C++ compiler and CMake, e.g.:
```
cd cpp/build
cmake .. -G "Unix Makefiles"
make
make install
```
Executables will be put in cpp/bin folder.


## Python examples

Required modules: pytesseract, tesserocr, cv2.


### References

[^1]: <https://github.com/tesseract-ocr/tesseract>
[^2]: <https://opencv.org/>
[^3]: <https://www.qt.io/>
