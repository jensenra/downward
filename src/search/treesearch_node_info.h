#ifndef TREESEARCH_NODE_INFO_H
#define TREESEARCH_NODE_INFO_H

#include "operator_id.h"
#include "state_id.h"
#include "search_node_info.h"

#include <vector>

using namespace std;

struct TreeSearchNodeInfo : public SearchNodeInfo{
    int l;
    int reward_sum;
    int visited;
    int best_h;
    vector<StateID> children_state_ids;
    vector<StateID> forgotten_children;
    

    TreeSearchNodeInfo();

    public:
    StateID get_parent();
    OperatorID get_operator();
    void remove_child(StateID id);
};

#endif
