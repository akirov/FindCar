#include <memory>
#include "TessTextReader.hpp"

using namespace imgtotxt;

TessTextReader::TessTextReader(const std::string& tessdata)
{
    m_tessAPI.Init(tessdata.c_str(), "eng");
    m_tessAPI.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_LINE);
}


TessTextReader::~TessTextReader()
{
    m_tessAPI.Clear();
    m_tessAPI.End();
}


std::string TessTextReader::imgToTxt(const uint8_t *imgData, int width, int height, int bytesPerPixel, int bytesPerLine)
{
    m_tessAPI.SetImage(imgData, width, height, bytesPerPixel, bytesPerLine);
    auto text = std::unique_ptr<char[]>{m_tessAPI.GetUTF8Text()};
    return std::string{text.get()};
}


std::string TessTextReader::imgToTxt(const std::string& imgURI)
{
    std::unique_ptr<Pix, PixDeleter> image{pixRead(imgURI.c_str())};
    if( !image ) return std::string{};
    m_tessAPI.SetImage(image.get());
    auto text = std::unique_ptr<char[]>{m_tessAPI.GetUTF8Text()};
    return std::string{text.get()};
}
