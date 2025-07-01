#include "RandomForest.hpp"
#include <algorithm>
#include <map>
#include <thread>

/**
 * Constructeur de la forêt aléatoire
 * @param n_trees Nombre d'arbres
 * @param max_depth Profondeur maximale
 * @param min_samples_split Nombre minimal d'échantillons pour diviser
 */
RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_split)
    : n_trees(n_trees), max_depth(max_depth), min_samples_split(min_samples_split) {}

/**
 * Entraîne la forêt sur les données X et les labels y
 * @param X Données d'entrée
 * @param y Labels
 */
void RandomForest::fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
    trees.clear();
    trees.resize(n_trees, DecisionTree(max_depth, min_samples_split));
    std::vector<std::thread> threads;
    for (int i = 0; i < n_trees; ++i) {
        threads.emplace_back([&, i]() {
            std::vector<int> y_sample;
            auto X_sample = bootstrap_sample(X, y_sample, y);
            trees[i].fit(X_sample, y_sample);
        });
    }
    for (auto& t : threads) t.join();
}

/**
 * Prédit la classe pour un échantillon x en utilisant le vote majoritaire des arbres
 * @param x Vecteur d'attributs
 * @return Classe prédite
 */
int RandomForest::predict(const std::vector<int>& x) const {
    std::map<int, int> votes;
    for (const auto& tree : trees) {
        int pred = tree.predict(x);
        votes[pred]++;
    }
    int majority = -1, max_count = 0;
    for (auto& kv : votes) {
        if (kv.second > max_count) {
            max_count = kv.second;
            majority = kv.first;
        }
    }
    return majority;
}

/**
 * Génère un échantillon bootstrapé à partir des données d'entrée
 * @param X Données d'entrée
 * @param y_sample Labels échantillonnés (rempli par la fonction)
 * @param y Labels originaux
 * @return Données échantillonnées
 */
std::vector<std::vector<int>> RandomForest::bootstrap_sample(const std::vector<std::vector<int>>& X, std::vector<int>& y_sample, const std::vector<int>& y) const {
    std::vector<std::vector<int>> X_sample;
    y_sample.clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, X.size() - 1);
    for (size_t i = 0; i < X.size(); ++i) {
        int idx = dis(gen);
        X_sample.push_back(X[idx]);
        y_sample.push_back(y[idx]);
    }
    return X_sample;
}
