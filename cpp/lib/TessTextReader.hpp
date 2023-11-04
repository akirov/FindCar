#ifndef __TESSTEXTREADER_HPP__
#define __TESSTEXTREADER_HPP__

#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

namespace imgtotxt {

class TessTextReader  // : public ITextReader
{
  public:
    TessTextReader(const std::string& tessdata="");
    ~TessTextReader();

    std::string imgToTxt(const uint8_t *imgData, int width, int height, int bytesPerPixel, int bytesPerLine);  // override
    std::string imgToTxt(const std::string& imgURI);  // override

    void setPageSegMode(tesseract::PageSegMode mode) { m_tessAPI.SetPageSegMode(mode); }

  private:
    tesseract::TessBaseAPI m_tessAPI;
};


struct PixDeleter
{
    void operator()(Pix *img) { pixDestroy(&img); }
};


}

#endif  // __TESSTEXTREADER_HPP__
