#include "treesearch_space.h"
#include "search_space.h"

#include "treesearch_node_info.h"
#include "task_proxy.h"

#include "task_utils/task_properties.h"
#include "utils/logging.h"

#include <cassert>

using namespace std;

TreeSearchNode::TreeSearchNode(const State &state, TreeSearchNodeInfo &info) :
    info(info), state(state){
    }
    const State &SearchNode::get_state() const {
    return state;
}

bool TreeSearchNode::is_open() const {
    return info.status == TreeSearchNodeInfo::OPEN;
}

bool TreeSearchNode::is_closed() const {
    return info.status == TreeSearchNodeInfo::CLOSED;
}

bool TreeSearchNode::is_dead_end() const {
    return info.status == TreeSearchNodeInfo::DEAD_END;
}

bool TreeSearchNode::is_new() const {
    return info.status == TreeSearchNodeInfo::NEW;
}

int TreeSearchNode::get_g() const {
    assert(info.g >= 0);
    return info.g;
}

int TreeSearchNode::get_real_g() const {
    return info.real_g;
}

vector<StateID> TreeSearchNode::get_children(){
    return info.children_state_ids;
}

void TreeSearchNode::add_child(StateID &childID){
    info.children_state_ids.push_back(childID);
}

void TreeSearchNode::open_initial() {
    assert(info.status == TreeSearchNodeInfo::NEW);
    info.status = TreeSearchNodeInfo::OPEN;
    info.g = 0;
    info.real_g = 0;
    info.parent_state_id = StateID::no_state;
    info.creating_operator = OperatorID::no_operator;
}

void TreeSearchNode::open(const TreeSearchNode &parent_node,
                      const OperatorProxy &parent_op,
                      int adjusted_cost) {
    assert(info.status == TreeSearchNodeInfo::NEW);
    info.status = TreeSearchNodeInfo::OPEN;
    info.g = parent_node.info.g + adjusted_cost;
    info.real_g = parent_node.info.real_g + parent_op.get_cost();
    info.parent_state_id = parent_node.get_state().get_id();
    info.creating_operator = OperatorID(parent_op.get_id());
}

void TreeSearchNode::reopen(const TreeSearchNode &parent_node,
                        const OperatorProxy &parent_op,
                        int adjusted_cost) {
    assert(info.status == TreeSearchNodeInfo::OPEN ||
           info.status == TreeSearchNodeInfo::CLOSED);

    // The latter possibility is for inconsistent heuristics, which
    // may require reopening closed nodes.
    info.status = TreeSearchNodeInfo::OPEN;
    info.g = parent_node.info.g + adjusted_cost;
    info.real_g = parent_node.info.real_g + parent_op.get_cost();
    info.parent_state_id = parent_node.get_state().get_id();
    info.creating_operator = OperatorID(parent_op.get_id());
}

// like reopen, except doesn't change status
void TreeSearchNode::update_parent(const TreeSearchNode &parent_node,
                               const OperatorProxy &parent_op,
                               int adjusted_cost) {
    assert(info.status == TreeSearchNodeInfo::OPEN ||
           info.status == TreeSearchNodeInfo::CLOSED);
    // The latter possibility is for inconsistent heuristics, which
    // may require reopening closed nodes.
    info.g = parent_node.info.g + adjusted_cost;
    info.real_g = parent_node.info.real_g + parent_op.get_cost();
    info.parent_state_id = parent_node.get_state().get_id();
    info.creating_operator = OperatorID(parent_op.get_id());
}

void TreeSearchNode::close() {
    assert(info.status == TreeSearchNodeInfo::OPEN);
    info.status = TreeSearchNodeInfo::CLOSED;
}

void TreeSearchNode::mark_as_dead_end() {
    info.status = TreeSearchNodeInfo::DEAD_END;
}

void TreeSearchNode::dump(const TaskProxy &task_proxy, utils::LogProxy &log) const {
    log << state.get_id() << ": ";
    task_properties::dump_fdr(state);
    if (info.creating_operator != OperatorID::no_operator) {
        OperatorsProxy operators = task_proxy.get_operators();
        OperatorProxy op = operators[info.creating_operator.get_index()];
        log << " created by " << op.get_name()
            << " from " << info.parent_state_id << endl;
    } else {
        log << " no parent" << endl;
    }
}

TreeSearchSpace::TreeSearchSpace(StateRegistry &state_registry, utils::LogProxy &log)
    : state_registry(state_registry), log(log) {
}

TreeSearchNode TreeSearchSpace::get_node(const State &state) {
    return TreeSearchNode(state, search_node_infos[state]);
}

void TreeSearchSpace::trace_path(const State &goal_state,
                             vector<OperatorID> &path) const {
    State current_state = goal_state;
    assert(current_state.get_registry() == &state_registry);
    assert(path.empty());
    for (;;) {
        const TreeSearchNodeInfo &info = search_node_infos[current_state];
        if (info.creating_operator == OperatorID::no_operator) {
            assert(info.parent_state_id == StateID::no_state);
            break;
        }
        path.push_back(info.creating_operator);
        current_state = state_registry.lookup_state(info.parent_state_id);
    }
    reverse(path.begin(), path.end());
}

void TreeSearchSpace::dump(const TaskProxy &task_proxy) const {
    OperatorsProxy operators = task_proxy.get_operators();
    for (StateID id : state_registry) {
        /* The body duplicates SearchNode::dump() but we cannot create
           a search node without discarding the const qualifier. */
        State state = state_registry.lookup_state(id);
        const TreeSearchNodeInfo &node_info = search_node_infos[state];
        log << id << ": ";
        task_properties::dump_fdr(state);
        if (node_info.creating_operator != OperatorID::no_operator &&
            node_info.parent_state_id != StateID::no_state) {
            OperatorProxy op = operators[node_info.creating_operator.get_index()];
            log << " created by " << op.get_name()
                << " from " << node_info.parent_state_id << endl;
        } else {
            log << "has no parent" << endl;
        }
    }
}

void SearchSpace::print_statistics() const {
    state_registry.print_statistics(log);
}
