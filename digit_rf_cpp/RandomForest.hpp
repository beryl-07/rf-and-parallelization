#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP
#include "DecisionTree.hpp"
#include <vector>
#include <random>

/**
 * Classe représentant une forêt aléatoire (Random Forest) pour la classification
 * Permet d'entraîner plusieurs arbres de décision et d'effectuer des prédictions par vote majoritaire
 * Constructeur de la forêt aléatoire
 * @param n_trees Nombre d'arbres
 * @param max_depth Profondeur maximale
 * @param min_samples_split Nombre minimal d'échantillons pour diviser
 */
class RandomForest {
public:
    /**
     * Constructeur de la forêt aléatoire
     * @param n_trees Nombre d'arbres dans la forêt
     * @param max_depth Profondeur maximale de chaque arbre
     * @param min_samples_split Nombre minimal d'échantillons pour diviser un noeud
     */
    RandomForest(int n_trees = 10, int max_depth = 8, int min_samples_split = 2);
    /**
     * Entraîne la forêt sur les données X et les labels y
     * @param X Données d'entrée (vecteurs d'attributs)
     * @param y Labels correspondants
     */
    void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y);
    /**
     * Prédit la classe pour un échantillon x en utilisant le vote majoritaire des arbres
     * @param x Vecteur d'attributs
     * @return Classe prédite
     */
    int predict(const std::vector<int>& x) const;
private:
    int n_trees; ///< Nombre d'arbres
    int max_depth; ///< Profondeur maximale
    int min_samples_split; ///< Nombre minimal d'échantillons pour diviser
    std::vector<DecisionTree> trees; ///< Liste des arbres de la forêt
    /**
     * Génère un échantillon bootstrapé à partir des données d'entrée
     * @param X Données d'entrée
     * @param y_sample Labels échantillonnés (rempli par la fonction)
     * @param y Labels originaux
     * @return Données échantillonnées
     */
    std::vector<std::vector<int>> bootstrap_sample(const std::vector<std::vector<int>>& X, std::vector<int>& y_sample, const std::vector<int>& y) const;
};

#endif // RANDOM_FOREST_HPP
