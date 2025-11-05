#include <iostream>
#include <string>
#include "filters.hpp"
#include "io.hpp"

using namespace filters;

static void usage(){
    std::cout << "image_filters -i in -o out -f grayscale|boxblur|gauss5|sobel|unsharp "
                 "[--device cpu|cuda] [--tile N] [--amount A]\n";
}

int main(int argc, char** argv) {
    std::string in, out, f = "grayscale";
    Device device = Device::CUDA;
    int tile = 16;
    int iters = 1;
    float amount = 1.0f;

    // parse arguments 
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]{ return (i+1<argc)? std::string(argv[++i]) : std::string(); };
        if (s=="-i"||s=="--in") in = next();
        else if (s=="-o"||s=="--out") out = next();
        else if (s=="-f"||s=="--filter") f = next();
        else if (s=="--device") device = (next()=="cpu"? Device::CPU : Device::CUDA);
        else if (s=="--tile") tile = std::stoi(next());
        else if (s=="--iters") iters = std::max(1, std::stoi(next()));
        else if (s=="--amount") amount = std::stof(next());
        else { usage(); return 1; }
    }
    if (in.empty()||out.empty()){ usage(); return 1; }

    Filter filter;
    if      (f=="grayscale") filter = Filter::Grayscale;
    else if (f=="boxblur")   filter = Filter::Boxblur;
    else if (f=="gauss")    filter = Filter::GaussianBlur;
    else if (f=="sobel")     filter = Filter::SobelEdge;
    else if (f=="unsharp")   filter = Filter::Unsharp;
    else { std::cerr<<"Unknown filter "<< f << "\n"; return 2; }

    ImageU8 img;
    if (!readPPM(in, img)) { 
        std::cerr<<"Failed to read "<<in<<"\n"; 
        return 3;
    }

    ImageU8 outImg;
    for (int k=0;k<iters;++k){
        if (filter == Filter::Unsharp && device == Device::CUDA){
            if (!unsharpCUDA(img, outImg, amount, tile, 0)){ std::cerr << "apply failed\n"; return 4; }
        } else {
            if (!apply(filter, device, img, outImg, tile, nullptr)){ std::cerr<<"apply failed\n"; return 4; }
        }
    }

    if (!writeImage(out, outImg)){ std::cerr<<"Write failed: "<<out<<"\n"; return 5; }
    std::cout<<"OK -> "<<out<<" ("<<outImg.w<<"x"<<outImg.h<<", c="<<outImg.c<<")\n";
    return 0;
}
