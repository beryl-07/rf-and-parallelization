#include "DecisionTree.hpp"
#include <algorithm>
#include <map>
#include <limits>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <sstream>
#include <cmath>

/**
 * Destructeur du noeud : libère récursivement les sous-arbres gauche et droit
 */
DecisionTree::Node::~Node() {
    delete left;
    delete right;
}

/**
 * Constructeur de l'arbre de décision
 * @param max_depth Profondeur maximale
 * @param min_samples_split Nombre minimal d'échantillons pour diviser
 */
DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split) {}

/**
 * Entraîne l'arbre de décision sur les données X et les labels y
 * @param X Données d'entrée
 * @param y Labels
 */
void DecisionTree::fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
    delete_tree(root);
    root = build(X, y, 0);
    // draw_tree(); // Affiche l'arbre après l'entraînement
    // // Génère un nom de fichier unique basé sur l'heure courante
    // namespace fs = std::filesystem;
    // fs::create_directories("trees/dot");
    // auto now = std::chrono::system_clock::now();
    // auto now_time = std::chrono::system_clock::to_time_t(now);
    // std::string base = "trees/dot/arbre_" + std::to_string(now_time);
    // std::string filename = base + ".dot";
    // // Si le fichier existe déjà, ajoute une chaîne aléatoire
    // if (fs::exists(filename)) {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_int_distribution<> dis(10000, 99999);
    //     std::stringstream ss;
    //     ss << base << "_" << dis(gen) << ".dot";
    //     filename = ss.str();
    // }
    // export_dot(filename);
}

/**
 * Prédit la classe pour un échantillon x
 * @param x Vecteur d'attributs
 * @return Classe prédite
 */
int DecisionTree::predict(const std::vector<int>& x) const {
    return predict_node(root, x);
}

/**
 * Supprime récursivement l'arbre
 * @param node Racine du sous-arbre à supprimer
 */
void DecisionTree::delete_tree(Node* node) {
    if (node) delete node;
}

/**
 * Retourne la classe majoritaire dans y
 * @param y Labels
 * @return Classe majoritaire
 */
int DecisionTree::majority_class(const std::vector<int>& y) const {
    std::map<int, int> counts;
    for (int label : y) counts[label]++;
    int max_count = 0, majority = -1;
    for (auto& kv : counts) {
        if (kv.second > max_count) {
            max_count = kv.second;
            majority = kv.first;
        }
    }
    return majority;
}

/**
 * Calcule l'impureté de Gini pour un vecteur de labels
 * @param y Labels
 * @return Impureté de Gini
 */
double DecisionTree::gini(const std::vector<int>& y) const {
    std::map<int, int> counts;
    for (int label : y) counts[label]++;
    double impurity = 1.0;
    int n = y.size();
    for (auto& kv : counts) {
        double p = (double)kv.second / n;
        impurity -= p * p;
    }
    return impurity;
}

/**
 * Construit récursivement l'arbre de décision
 * @param X Données d'entrée
 * @param y Labels
 * @param depth Profondeur actuelle
 * @return Pointeur vers le noeud racine du sous-arbre
 */
DecisionTree::Node* DecisionTree::build(const std::vector<std::vector<int>> &X, const std::vector<int> &y, int depth) {
    if (y.empty()) return nullptr;
    if (depth >= max_depth || y.size() < static_cast<std::vector<int>::size_type>(min_samples_split) || gini(y) == 0.0) {
        Node* leaf = new Node();
        leaf->label = majority_class(y);
        return leaf;
    }
    int n_features = X[0].size();
    int n_sub_features = std::max(1, static_cast<int>(std::sqrt(n_features)));
    std::vector<int> feature_indices(n_features);
    for (int i = 0; i < n_features; ++i) feature_indices[i] = i;
    // Mélange aléatoire des indices de features
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    // On ne considère qu'un sous-ensemble aléatoire de features
    int best_feat = -1, best_val = 0;
    double best_gini = std::numeric_limits<double>::max();
    std::vector<int> left_idx, right_idx;
    for (int f = 0; f < n_sub_features; ++f) {
        int feat = feature_indices[f];
        for (int val = 0; val <= 1; ++val) {
            std::vector<int> l, r;
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][feat] == val) l.push_back(i);
                else r.push_back(i);
            }
            if (l.empty() || r.empty()) continue;
            std::vector<int> y_left, y_right;
            for (int idx : l) y_left.push_back(y[idx]);
            for (int idx : r) y_right.push_back(y[idx]);
            double g = (y_left.size() * gini(y_left) + y_right.size() * gini(y_right)) / y.size();
            if (g < best_gini) {
                best_gini = g;
                best_feat = feat;
                best_val = val;
                left_idx = l;
                right_idx = r;
            }
        }
    }
    if (best_feat == -1) {
        Node* leaf = new Node();
        leaf->label = majority_class(y);
        return leaf;
    }
    std::vector<std::vector<int>> X_left, X_right;
    std::vector<int> y_left, y_right;
    for (int idx : left_idx) {
        X_left.push_back(X[idx]);
        y_left.push_back(y[idx]);
    }
    for (int idx : right_idx) {
        X_right.push_back(X[idx]);
        y_right.push_back(y[idx]);
    }
    Node* node = new Node();
    node->feature = best_feat;
    node->value = best_val;
    node->left = build(X_left, y_left, depth + 1);
    node->right = build(X_right, y_right, depth + 1);
    return node;
}

/**
 * Prédit la classe pour un échantillon x à partir d'un noeud donné
 * @param node Noeud courant
 * @param x Vecteur d'attributs
 * @return Classe prédite
 */
int DecisionTree::predict_node(const Node* node, const std::vector<int>& x) const {
    if (!node->left && !node->right) return node->label;
    if (x[node->feature] == node->value) {
        return predict_node(node->left, x);
    } else {
        return predict_node(node->right, x);
    }
}