#ifndef SEARCH_TREE_H
#define SEARCH_TREE_H

#include "search_space.h"
#include "search_engine.h"
#include "state_registry.h"

class SearchNode;

class TreeSearchNode{
    SearchNode current;
    SearchNode children[];
    public:
    TreeSearchNode(SearchNode &current, SearchNode children[]);
};

class SearchTree{
    SearchNode root;

    public:
    SearchTree(const SearchNode &root);
    void expand(SearchNode &node);
    void close(SearchNode &node);
    void dead_end(SearchNode &node);
    void step();
};

#endif