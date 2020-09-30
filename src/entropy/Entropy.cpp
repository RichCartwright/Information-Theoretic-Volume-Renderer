
#include <cstdio>
#include <iostream>

#include "Entropy.h"

void Entropy::GetEntropy(   uint* histA, uint* histB, size_t binCount, 
                            float* entA, float* entB, 
                            float* jEnt, float* cEnt, 
                            float* mI,
                            int* sum)
{
// Entropy
    Eigen::MatrixXd eigHistA;
    Eigen::Map<Eigen::MatrixXi> mapHistA((int*)histA, 1, binCount);
    eigHistA = mapHistA.cast<double>();
    eigHistA = eigHistA / eigHistA.sum();
    //GetNonZero(&eigHistA); 

    Eigen::MatrixXd eigHistB;
    if(histB != nullptr) 
    {
        Eigen::Map<Eigen::MatrixXi> mapHistB((int*)histB, 1, binCount);
        eigHistB = mapHistB.cast<double>();
        eigHistB = eigHistB / eigHistB.sum();
        //GetNonZero(&eigHistB);
        *entB = eigHistB.cwiseProduct(Log2(eigHistB)).sum() * -1;
    }

    *entA = eigHistA.cwiseProduct(Log2(eigHistA)).sum() * -1;

// Joint Entropy -  The size has to be the same, so its best not to take the non-zero matrix,
//                  we just need to check for zero when taking the log.
    if(eigHistB.size() == eigHistA.size())
    {
        Eigen::MatrixXd eigJointHist; 
        eigJointHist = eigHistA + eigHistB; 
        *jEnt = eigJointHist.cwiseProduct(Log2(eigJointHist)).sum() * -1;
        *mI = *entA + *entB - *jEnt;
    }
    
    // Set the total sum of the histogram data
    *sum = GetTotal(histA, binCount);
}

float Entropy::SingleEntropy(uint* hist, size_t bin_count)
{
    Eigen::MatrixXd eigHist;
    Eigen::Map<Eigen::MatrixXi> mapHist((int*)hist, 1, bin_count);
    eigHist = mapHist.cast<double>();
    eigHist = eigHist / eigHist.sum();
    GetNonZero(&eigHist); 

    return eigHist.cwiseProduct(Log2(eigHist)).sum() * -1;
}

void Entropy::GetNonZero(Eigen::MatrixXd* inMat)
{
    Eigen::Matrix<bool, Eigen::Dynamic, 1> zeros = (inMat->array() == 0).colwise().all();

    size_t last = inMat->cols() - 1;
    for(size_t i = 0; i < last + 1;)
    {
        if(zeros(i))
        {
            inMat->col(i).swap(inMat->col(last));
            zeros.segment<1>(i).swap(zeros.segment<1>(last)); 
            --last; 
        }
        else{ ++i; }
    }
    inMat->conservativeResize(inMat->rows(), last+1);
}

double Entropy::StandardDeviation(Eigen::MatrixXd inMat)
{
    if(inMat.size() <= 0.0)
    {
        fprintf(stderr, "Entropy::StandardDeviation(): Size <= 0 matrix passed in "); 
        return 0.0;
    }

    double mean = inMat.mean();
    for(int i = 0; i < inMat.size(); ++i)
    {
        inMat(i) -= mean;
        inMat(i) = powf(inMat(i), 2);
    }

    return inMat.sum() / powf(inMat.size(), 2);
}


Eigen::MatrixXd Entropy::Log2(Eigen::MatrixXd inMat)
{
    for(int i = 0; i < inMat.size(); ++i)
    {
        if(inMat(i) != 0)
        {
            inMat(i) = std::log(inMat(i)) / std::log(2); 
        }
        //inMat(i) = std::log2f(inMat(i)); 
    }
    return inMat;
}

int Entropy::GetTotal(uint* inHist, size_t bin_count)
{
    int sum = 0;
    for(size_t i = 0; i < bin_count; ++i)
    {
        sum += inHist[i]; 
    }
    return sum;
}
