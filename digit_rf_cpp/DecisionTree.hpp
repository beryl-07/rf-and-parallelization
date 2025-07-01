#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP
#include <vector>
#include <cstddef>
#include <string>
#include <ostream>

// Classe représentant un arbre de décision binaire pour la classification
// Cette classe permet d'entraîner un arbre de décision et de prédire la classe d'un échantillon
class DecisionTree {
public:
    /**
     * Constructeur de l'arbre de décision
     * @param max_depth Profondeur maximale de l'arbre
     * @param min_samples_split Nombre minimal d'échantillons pour diviser un noeud
     */
    DecisionTree(int max_depth = 5, int min_samples_split = 2);
    /**
     * Entraîne l'arbre de décision sur les données X et les labels y
     * @param X Données d'entrée (vecteurs d'attributs)
     * @param y Labels correspondants
     */
    void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y);
    /**
     * Prédit la classe pour un échantillon x
     * @param x Vecteur d'attributs
     * @return Classe prédite
     */
    int predict(const std::vector<int>& x) const;

private:
    /**
     * Structure représentant un noeud de l'arbre
     */
    struct Node {
        int feature = -1; ///< Index de la caractéristique utilisée pour la division
        int value = 0;    ///< Valeur de la caractéristique pour la division
        int label = -1;   ///< Classe prédite si feuille
        Node* left = nullptr;  ///< Fils gauche
        Node* right = nullptr; ///< Fils droit
        /**
         * Destructeur du noeud (libère les sous-arbres)
         */
        ~Node();
    };
    Node* root; ///< Racine de l'arbre
    int max_depth; ///< Profondeur maximale
    int min_samples_split; ///< Nombre minimal d'échantillons pour diviser
    /**
     * Construit récursivement l'arbre de décision
     * @param X Données d'entrée
     * @param y Labels
     * @param depth Profondeur actuelle
     * @return Pointeur vers le noeud racine du sous-arbre
     */
    Node* build(const std::vector<std::vector<int>> &X, const std::vector<int> &y, int depth);
    /**
     * Retourne la classe majoritaire dans y
     * @param y Labels
     * @return Classe majoritaire
     */
    int majority_class(const std::vector<int>& y) const;
    /**
     * Calcule l'impureté de Gini pour un vecteur de labels
     * @param y Labels
     * @return Impureté de Gini
     */
    double gini(const std::vector<int>& y) const;
    /**
     * Supprime récursivement l'arbre
     * @param node Racine du sous-arbre à supprimer
     */
    void delete_tree(Node* node);
    /**
     * Prédit la classe pour un échantillon x à partir d'un noeud donné
     * @param node Noeud courant
     * @param x Vecteur d'attributs
     * @return Classe prédite
     */
    int predict_node(const Node* node, const std::vector<int>& x) const;
    
};

#endif // DECISION_TREE_HPP
