#pragma once
#include <stdint.h>
#include <thread>
#include <cmath>

namespace NNCpp
{

class XorShift {
    uint32_t xorsft_x;
    uint32_t xorsft_y;
    uint32_t xorsft_z;
    uint32_t xorsft_w;
public:
    XorShift() {
        xorsft_x = 123456789;
        xorsft_y = 362436069;
        xorsft_z = 521288629;
        xorsft_w = 88675123;
    }

    void initrand(uint32_t seed){
        do {
            seed = seed*1812433253 + 1; seed ^= seed<<13; seed ^= seed>>17;
            xorsft_x = 123464980 ^ seed;
            seed = seed*1812433253 + 1; seed ^= seed<<13; seed ^= seed>>17;
            xorsft_y = 3447902351 ^ seed;
            seed = seed*1812433253 + 1; seed ^= seed<<13; seed ^= seed>>17;
            xorsft_z = 2859490775 ^ seed;
            seed = seed*1812433253 + 1; seed ^= seed<<13; seed ^= seed>>17;
            xorsft_w = 47621719 ^ seed;
        } while(xorsft_x==0 && xorsft_y==0 && xorsft_z==0 && xorsft_w==0);
    }

    double urand(){
        uint32_t t;
        t = xorsft_x ^ (xorsft_x<<11);
        xorsft_x = xorsft_y;
        xorsft_y = xorsft_z;
        xorsft_z = xorsft_w;
        xorsft_w ^= t ^ (t>>8) ^ (xorsft_w>>19);
        return ((xorsft_x+0.5) / 4294967296.0 + xorsft_w) / 4294967296.0;
    }

    double operator() () {
        return urand();
    }

    double normal() {
        double x = urand(), y = urand();
        return sqrt(-2 * log(x)) * cos(2*acos(-1)*y);
    }
};

}