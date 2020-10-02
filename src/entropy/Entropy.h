#ifndef HEMELB_ENTROPY_H
#define HEMELB_ENTROPY_H

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

class Entropy {

    public:
        static Entropy *getInstance()
        {
            if(!instance){
                instance = new Entropy;
            }
            return instance;
        }

        void GetEntropy(uint* histA, uint* histB, size_t binCount, float* entA, float* entB, float* jEnt, float* mI);
        float SingleEntropy(uint* hist, size_t bin_count);


    private:
        static Entropy *instance; 
        Entropy(){};

        void             GetNonZero(Eigen::MatrixXd* inMat);
        double           StandardDeviation(Eigen::MatrixXd inMat);
        Eigen::MatrixXd  Log2(Eigen::MatrixXd inMat);
        int              GetTotal(uint* inMat, size_t bin_count);
};
#endif