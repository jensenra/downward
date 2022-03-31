#ifndef TREESEARCH_NODE_INFO_H
#define TREESEARCH_NODE_INFO_H

#include "operator_id.h"
#include "state_id.h"
#include "search_node_info.h"

#include <vector>

using namespace std;

struct TreeSearchNodeInfo : public SearchNodeInfo{
    vector<StateID> children_state_ids;

    TreeSearchNodeInfo();
};

#endif
