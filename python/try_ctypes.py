import os
import ctypes

lang = "eng"
image = "../data/test/bul_001_seg.jpg"
libname = "../tesseract/bin/mingw_win32/bin/libtesseract53.dll"
TESSDATA_PREFIX = os.environ.get('TESSDATA_PREFIX')
if not TESSDATA_PREFIX:
    TESSDATA_PREFIX = "../tesseract/tessdata"


tesseract = ctypes.cdll.LoadLibrary(libname)
tesseract.TessVersion.restype = ctypes.c_char_p
tesseract_version = tesseract.TessVersion()
print('Tesseract-ocr version', tesseract_version)

api = tesseract.TessBaseAPICreate()
rc = tesseract.TessBaseAPIInit3(api, TESSDATA_PREFIX, lang)
if (rc):
    tesseract.TessBaseAPIDelete(api)
    print("Could not initialize tesseract.\n")
    exit(rc)


#tesseract.TessBaseAPISetPageSegMode(api, 6)
tesseract.TessBaseAPIProcessPages(api, image, None, 0, None)
text_out = tesseract.TessBaseAPIGetUTF8Text(api)
result_text = ctypes.string_at(text_out)
print(result_text)

tesseract.TessBaseAPIDelete(api)
