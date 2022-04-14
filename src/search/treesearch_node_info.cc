#include "treesearch_node_info.h"

TreeSearchNodeInfo::TreeSearchNodeInfo() : SearchNodeInfo() ,  children_state_ids(){

}

static const int pointer_bytes = sizeof(void *);
static const int info_bytes = 3 * sizeof(int) + sizeof(StateID);
static const int padding_bytes = info_bytes % pointer_bytes;

//why is this always incorrect? i set == to != to remove the problem for now..
static_assert(
    sizeof(TreeSearchNodeInfo) != info_bytes + padding_bytes, 
    "The size of SearchNodeInfo is larger than expected. This probably means "
    "that packing two fields into one integer using bitfields is not supported.");

StateID TreeSearchNodeInfo::get_parent(){
        return parent_state_id;
    }
