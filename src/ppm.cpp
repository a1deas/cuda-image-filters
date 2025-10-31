#include "ppm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cctype>

using namespace filters;

static void skipWScomments(FILE* f){
  int ch;
  do {
    ch=fgetc(f);
    if(ch=='#'){
      while(ch!='\n' && ch!=EOF) ch=fgetc(f);
    }
  } while(isspace(ch));
  ungetc(ch,f);
}

bool filters::readPPM(const std::string& path, ImageU8& img){
  FILE* f = std::fopen(path.c_str(), "rb");
  if(!f) return false;

  char magic[3]{};
  if(std::fread(magic,1,2,f)!=2 || magic[0]!='P' || magic[1]!='6') {
    std::fclose(f); return false;
  }

  skipWScomments(f);
  int w,h,maxv;
  if(std::fscanf(f, "%d", &w)!=1){ std::fclose(f); return false; }
  skipWScomments(f);
  if(std::fscanf(f, "%d", &h)!=1){ std::fclose(f); return false; }
  skipWScomments(f);
  if(std::fscanf(f, "%d", &maxv)!=1 || maxv!=255){ std::fclose(f); return false; }

  fgetc(f);

  img.w=w; img.h=h; img.c=3;
  img.data.resize((size_t)w*h*3);
  size_t n = std::fread(img.data.data(), 1, img.data.size(), f);
  std::fclose(f);
  return n == img.data.size();
}

bool filters::writePPM(const std::string& path, const ImageU8& img){
  if(!(img.c==1 || img.c==3)) return false;
  FILE* f = std::fopen(path.c_str(), "wb"); if(!f) return false;
  std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
  if(img.c==3) {
    std::fwrite(img.data.data(), 1, img.data.size(), f);
  } else {
    for(int i=0;i<img.w*img.h;i++){
      unsigned char g = img.data[i];
      unsigned char rgb[3]{g,g,g};
      std::fwrite(rgb, 1, 3, f);
    }
  }
  std::fclose(f);
  return true;
}
