#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "TessTextReader.hpp"

// Use local tesseract data, if TESSDATA_PREFIX env variable is not set
#define LOCAL_TESSDATA "../../../tesseract/tessdata"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " image_uri" << std::endl;
        exit(-1);
    }
    std::string imageURI{argv[1]};

    if (! std::getenv("TESSDATA_PREFIX"))
    {
#ifdef __MINGW32__
        putenv("TESSDATA_PREFIX=" LOCAL_TESSDATA);
#else
        setenv("TESSDATA_PREFIX", LOCAL_TESSDATA, 1);
#endif // __MINGW32__
    }

    imgtotxt::TessTextReader textReader{};  // {TESSDATA_PREFIX}

    cv::Mat img = cv::imread(imageURI, cv::IMREAD_COLOR);
    std::string txt = textReader.imgToTxt(img.data, img.cols, img.rows, 3, img.step);
    std::cout << "Text: '" << txt << "'" << std::endl;

    txt = textReader.imgToTxt(imageURI);
    std::cout << "Text: '" << txt << "'" << std::endl;

    return 0;
}
