#include "RandomForest.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>

/**
 * Fonction principale : génère les jeux de données, entraîne la forêt aléatoire et affiche la précision
 * @return 0 si succès
 */
int main() {
    // Charger le dataset depuis un fichier CSV
    std::vector<std::vector<int>> X;
    std::vector<int> y;
    std::ifstream file("dataset/dataset.csv");
    if (!file) {
        std::cerr << "Erreur : impossible d'ouvrir dataset/dataset.csv" << std::endl;
        return 1;
    }
    std::string line;
    std::getline(file, line); // ignorer l'en-tête
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> features;
        std::string val;
        for (int i = 0; i < 7; ++i) {
            std::getline(ss, val, ',');
            features.push_back(std::stoi(val));
        }
        std::getline(ss, val, ',');
        y.push_back(std::stoi(val));
        X.push_back(features);
    }
    // Mélanger les données
    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    std::vector<std::vector<int>> X_shuffled;
    std::vector<int> y_shuffled;
    for (size_t i : indices) {
        X_shuffled.push_back(X[i]);
        y_shuffled.push_back(y[i]);
    }
    
    // Split 80% train, 20% test
    size_t n_train = X_shuffled.size() * 0.8;
    std::vector<std::vector<int>> X_train(X_shuffled.begin(), X_shuffled.begin() + n_train);
    std::vector<int> y_train(y_shuffled.begin(), y_shuffled.begin() + n_train);
    std::vector<std::vector<int>> X_test(X_shuffled.begin() + n_train, X_shuffled.end());
    std::vector<int> y_test(y_shuffled.begin() + n_train, y_shuffled.end());

    // Entraîner le Random Forest
    RandomForest rf(30, 5, 2);
    rf.fit(X_train, y_train);

    // Tester
    int correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        int pred = rf.predict(X_test[i]);
        if (pred == y_test[i]) ++correct;
        std::cout << "Vrai: " << y_test[i] << ", Prédit: " << pred << std::endl;
    }
    std::cout << "Précision: " << (double)correct / X_test.size() << std::endl;
    return 0;
}
