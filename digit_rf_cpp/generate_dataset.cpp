#include <iostream>
#include <fstream>
#include <vector>
#include <random>

// Génère un chiffre à 7 segments (vecteur binaire) avec du bruit
std::vector<int> generate_digit(int digit, double noise=0.1) {
    static const int segments[10][7] = {
        {1,1,1,0,1,1,1}, // 0
        {0,0,1,0,0,1,0}, // 1
        {1,0,1,1,1,0,1}, // 2
        {1,0,1,1,0,1,1}, // 3
        {0,1,1,1,0,1,0}, // 4
        {1,1,0,1,0,1,1}, // 5
        {1,1,0,1,1,1,1}, // 6
        {1,0,1,0,0,1,0}, // 7
        {1,1,1,1,1,1,1}, // 8
        {1,1,1,1,0,1,1}  // 9
    };
    std::vector<int> v(7);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution flip(noise);
    for (int i = 0; i < 7; ++i) {
        v[i] = segments[digit][i];
        if (flip(gen)) v[i] = 1 - v[i];
    }
    return v;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <n_samples> <noise> <output.csv>" << std::endl;
        return 1;
    }
    int n_samples = std::stoi(argv[1]);
    double noise = std::stod(argv[2]);
    std::string filename = argv[3];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> digit_dist(0, 9);
    std::ofstream out(filename);
    // En-tête CSV
    out << "s0,s1,s2,s3,s4,s5,s6,label\n";
    for (int i = 0; i < n_samples; ++i) {
        int d = digit_dist(gen);
        std::vector<int> v = generate_digit(d, noise);
        for (size_t j = 0; j < v.size(); ++j) {
            out << v[j];
            if (j < v.size() - 1) out << ",";
        }
        out << "," << d << "\n";
    }
    out.close();
    std::cout << "Dataset saved to " << filename << std::endl;
    return 0;
}
