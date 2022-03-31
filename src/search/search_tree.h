#ifndef SEARCH_TREE_H
#define SEARCH_TREE_H

#include "treesearch_space.h"
#include "search_space.h"
#include "search_engine.h"
#include "state_registry.h"

class SearchNode;

class SearchTree{
    TreeSearchNode root;

    public:
    SearchTree(const TreeSearchNode &root);
    void expand(SearchNode &node);
    void close(SearchNode &node);
    void dead_end(SearchNode &node);
    void step();
};

#endif