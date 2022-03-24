#include "search_tree.h"

using namespace std;

namespace search_tree {
    SearchTree::SearchTree(const SearchNode &node)
        : root(node){assert(&node != nullptr);}
    void SearchTree::expand(SearchNode &node){
    };
    void SearchTree::close(SearchNode &node){
    };
    void SearchTree::dead_end(SearchNode &node){
    };
    void SearchTree::step(){
    };
}