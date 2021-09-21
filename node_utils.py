"""

Divergence metric between two scores based on size of subgraph isomorphism. If
two DAGs are the exact same, the subgraph isomorphism will be of maximum size
and node divergence and edge divergence will be zero.

"""
import sys
import os
import json
import argparse

import numpy as np
import networkx as nx


def get_flow(f):
    return json.load(f)["flow"]


def simplify_flow(flow):
    if isinstance(flow, str):
        flow = json.loads(flow)

    s_flow = {}
    for node in flow:
        s_node = dict(**node)
        s_node["wires"] = s_node.get("wires", [])
        if len(s_node["wires"]) > 0 and isinstance(s_node["wires"][0], list):
            s_node["wires"] = sum(s_node["wires"], [])
        s_flow[s_node["id"]] = s_node
    return s_flow


def num_nodes(flow):
    return len(flow.keys())


def num_edges(flow):
    return sum(len(v["wires"]) for v in flow.values())


def has_edge(flow, k1, k2):
    return k2 in flow[k1]["wires"]


def edge_to_string(k1, k2):
    return " --> ".join([k1, k2])


def string_to_edge(s):
    return tuple(s.split(" --> "))


def get_node_similarity(node1, node2):
    if node1["type"] != node2["type"]:
        return 0

    # TODO: Fill this later if necessary #
    __skip_compares__ = set(
        [
            "id",
            "x",
            "y",
            "z",
            "wires",
            "type",
            "endpointUrl",  # for bot-intent type nodes
            "name",  # for ui_ nodes
            "group",  # for ui_ nodes
            "tab",  # for all nodes
            "label",  # for tab nodes
        ]
    )

    num = 0
    den = 0
    inc = 0
    for x in node1.keys():
        if x in __skip_compares__:
            continue

        # TODO:
        # skip node1[x] is str and matches ID regex:
        #
        # skip if attrs have a default_factory and don't need to be compared

        den += 1
        inc = 1
        val1 = node1.get(x, None)
        val2 = node2.get(x, None)
        if (val1 is None) ^ (val2 is None):
            # print(type(node1).__name__, "failed at", x)
            inc = 0
        elif not isinstance(val1, type(val1)) and not isinstance(val1, type(val2)):
            # print(type(node1).__name__, "failed at", x)
            inc = 0
        elif val1 != val2:
            # print(type(node1).__name__, "failed at", x)
            inc = 0

        num += inc

    # print(node1["type"], f"{num}/{den} properties match")
    if den == 0 or num == den:
        return 1
    else:
        return num / den


def mapping_weight(node1, node2):
    # only makes sense to compare nodes of the same type
    # can add additional conditions here if needed
    try:
        mnode1 = {k: v for k, v in node1.items() if k != "wires"}
        mnode2 = {k: v for k, v in node2.items() if k != "wires"}
        ans = get_node_similarity(mnode1, mnode2)
    except Exception as e:
        print("Comparison Exception:", e)
        print(
            "comparing",
            json.dumps(node1, indent=2),
            "\nand\n",
            json.dumps(node2, indent=2),
        )
        ans = 0
    return ans


def get_nodemap(flow1, flow2):
    nodemap = []
    for k1, v1 in flow1.items():
        for k2, v2 in flow2.items():
            wt = mapping_weight(v1, v2)
            if wt > 0:
                nodemap.append((k1, k2, wt))
    nodemap.sort(key=lambda x: (len(flow1[x[0]]["wires"]) + len(flow2[x[1]]["wires"])))
    return nodemap


def create_product_graph(nmap, flow1, flow2):
    prodgraph = set()

    for k1a, k2a, wta in nmap:
        for k1b, k2b, wtb in nmap:
            # assert one-to-one mapping
            if k1a == k1b or k2a == k2b:
                continue

            # is there is an edge between the two nodes in flow1?
            e_a = has_edge(flow1, k1a, k1b)

            # is there is an edge between the corresponding two nodes in flow2?
            e_b = has_edge(flow2, k2a, k2b)

            if not (e_a ^ e_b):
                # if (k1a, k1b) ⇔  (k2a, k2b), AND
                # the mapped nodes are of the same type,
                # add edge to product graph
                ind1 = nmap.index((k1a, k2a, wta))
                ind2 = nmap.index((k1b, k2b, wtb))
                edge = (min(ind1, ind2), max(ind1, ind2))
                prodgraph.add(edge)

    return list(prodgraph)


def density(pgraph, nmap):
    return (2 * len(pgraph)) / (len(nmap) * (len(nmap) - 1))


def check_clique(pgraph, clq):
    for i in clq:
        for j in clq:
            if (i != j) and (i, j) not in pgraph:
                return False

    return True


def large_graph_corr(pgraph, nmap, flow1, flow2):
    pg_arr = np.array(pgraph, dtype=np.uint64) + 1
    # runtime error if vertex numbers has 0, so add 1 and subtract when finding subset
    import cliquematch

    G = cliquematch.Graph.from_edgelist(pg_arr, len(nmap))

    exact = True
    dens = density(pgraph, nmap)
    if dens > 0.7:
        # highly dense graphs => node mapping is not strict enough,
        # (too many nodes of same type) so computing the exact value is SLOW
        # hence approximate via heuristic (some form of penalty)
        clique0 = G.get_max_clique(use_heuristic=True, use_dfs=False)
        # note that the approximate clique is <= the exact clique
        exact = False
    else:
        clique0 = G.get_max_clique(use_heuristic=True, use_dfs=True)

    clique = max(
        G.all_cliques(size=len(clique0)), key=setup_weighted_clique(nmap, flow1, flow2)
    )
    subset = [nmap[i - 1] for i in clique]
    return subset, exact


def setup_weighted_clique(nmap, flow1, flow2):
    def clique_wt(clq):
        wts = [nmap[x - 1][2] for x in clq]
        return sum(wts)

    return clique_wt


def small_graph_corr(pgraph, nmap, flow1, flow2):
    G = nx.Graph()
    G.add_nodes_from(i + 1 for i in range(len(nmap)))
    G.add_edges_from([(a + 1, b + 1) for a, b in pgraph])
    clique = max(
        nx.algorithms.clique.find_cliques(G),
        key=setup_weighted_clique(nmap, flow1, flow2),
    )
    subset = [nmap[x - 1] for x in clique]
    return subset, True


def find_correspondence(pgraph, nmap, flow1, flow2):
    if len(pgraph) == 0 and len(nmap) == 0:
        return [], True
    elif len(pgraph) < 2000:
        return small_graph_corr(pgraph, nmap, flow1, flow2)
    else:
        return large_graph_corr(pgraph, nmap, flow1, flow2)


def get_mapped_edges(subset, flow1, flow2):
    mapped_edges = {}
    for k1a, k2a, wta in subset:
        for k1b, k2b, wtb in subset:
            if k1a == k1b or k2a == k2b:
                continue
            # is there is an edge between the two nodes in flow1?
            e_a = has_edge(flow1, k1a, k1b)

            # is there is an edge between the corresponding two nodes in flow2?
            e_b = has_edge(flow2, k2a, k2b)

            if e_a and e_b:
                # successfully mapped the edge
                mapped_edges[edge_to_string(k1a, k1b)] = edge_to_string(k2a, k2b)
    return mapped_edges


def edge_similarity(edgemap, nodemap, flow1, flow2):
    if num_edges(flow1) != 0 and num_edges(flow2) != 0:
        return (len(edgemap) / num_edges(flow1)) * (len(edgemap) / num_edges(flow2))
    else:
        return 0


def node_similarity(subset, nodemap, flow1, flow2):
    if num_nodes(flow1) != 0 and num_nodes(flow2) != 0:
        score = sum(x[2] for x in subset)
        answer = (score / num_nodes(flow1)) * (score / num_nodes(flow2))
        return answer
    else:
        return 0


def get_divergence(full1, full2, edges_only=True):
    flow1 = simplify_flow(full1)
    flow2 = simplify_flow(full2)
    nmap = get_nodemap(flow1, flow2)
    pg = create_product_graph(nmap, flow1, flow2)
    corr, exact = find_correspondence(pg, nmap, flow1, flow2)
    emap = get_mapped_edges(corr, flow1, flow2)
    # print(f"{num_nodes(flow1)} nodes, {num_edges(flow1)} edges in flow1")
    # print(f"{num_nodes(flow2)} nodes, {num_edges(flow2)} edges in flow2")
    # print(len(emap), "edges mapped")
    ns = node_similarity(corr, nmap, flow1, flow2)
    es = edge_similarity(emap, nmap, flow1, flow2)

    if edges_only:
        return 1 - es
    else:
        return (1 - ns, 1 - es, exact)


def node_divergence(flow1, flow2):
    return get_divergence(flow1, flow2, False)[0]


def edge_divergence(flow1, flow2):
    return get_divergence(flow1, flow2, True)[1]


def runner(file1, file2):
    divergence = get_divergence(get_flow(file1), get_flow(file2), False)[0]
    return divergence


def main():
    parser = argparse.ArgumentParser(
        description="compare two JSON flows and return a divergence score"
    )
    parser.add_argument(
        "file1",
        type=argparse.FileType(mode="r"),
        help="path to JSON file 1. must have flow attribute",
    )
    parser.add_argument(
        "file2",
        type=argparse.FileType(mode="r"),
        help="path to JSON file 2. must have flow attribute",
    )

    d = parser.parse_args()
    print(runner(d.file1, d.file2))


if __name__ == "__main__":
    main()
