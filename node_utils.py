"""

Divergence metric between two scores based on size of subgraph isomorphism. If
two DAGs are the exact same, the subgraph isomorphism will be of maximum size
and node divergence and edge divergence will be zero.

"""
import json
import re
from random import randrange
from typing import List

import networkx as nx
from pydantic import BaseModel, Field, validator

GENERATE_NODEID = lambda: "%08x.fed%03x" % (
    randrange(16 ** 8),
    randrange(16 ** 3),
)
NODEID_PATTERN = "[0-9A-Fa-f]{6,8}\.[0-9A-Fa-f]{4,7}"
NODEID_REGEX = re.compile(NODEID_PATTERN)


def get_flow(f): return json.load(f)["flow"]

class NodeID(str):
    """
    Format of a node ID is:

    8-digit hex value (dot) 6-digit hex value
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            pattern=NODEID_PATTERN, examples=["abcdef01.234567", "01234567.fedcba"]
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError("NodeID requires string")
        v = v.lower().strip()[:15]

        # for compatibility with older JSON outputs
        # which occasionally have z as empty string
        if v == "":
            return GENERATE_NODEID()

        m = NODEID_REGEX.fullmatch(v)
        if not m:
            raise ValueError(f"Invalid NodeID format: {v}")

        return cls(m.group(0))


class _Node(BaseModel):
    __skip_compares__ = ("id", "type", "wires", "x", "y", "z")

    id: NodeID = Field(
        default_factory=GENERATE_NODEID,
        allow_mutation=False,
    )
    type: str = Field(default="", strip_whitespace=True)
    z: NodeID = Field(default="deadface.beefed")
    x: float = 0
    y: float = 0
    wires: List[List[NodeID]] = []

    class Config:
        title = "Node"
        extra = "ignore" 
        validate_assignment = True
        validate_all = True

    @validator("wires")
    def _check_wires(cls, v):
        if not isinstance(v, list):
            raise TypeError("expecting list for wires")

        ans = []
        if len(v) > 0:
            if isinstance(v[0], list):
                for sg in v:
                    assert isinstance(sg, list), "wires should be List[List]"
                    for x in sg:
                        NodeID.validate(x)
                ans = v
            elif isinstance(v[0], str):
                for sg in v:
                    assert isinstance(sg, str), "all elements should be str"
                    NodeID.validate(sg)
                ans = [v]
        return ans
    
    def similarity(self, other) -> float:
        if self.type != other.type:
            return 0

        num = 0
        den = 0
        inc = 0
        for x in self.__dict__.keys():
            if x in self.__skip_compares__:
                continue
            if x in type(self).__fields__ and type(self).__fields__[x].type_ == NodeID:
                continue
            if (
                x in type(self).__fields__
                and type(self).__fields__[x].default_factory is not None
            ):
                continue

            den += 1
            inc = 1
            val1 = getattr(self, x, None)
            val2 = getattr(other, x, None)
            if (val1 is None) ^ (val2 is None):
                inc = 0
            elif not isinstance(val1, type(val1)) and not isinstance(val1, type(val2)):
                inc = 0
            elif val1 != val2:
                inc = 0
            num += inc
        if den == 0 or num == den:
            return 1
        else:
            return num / den

    def __eq__(self, other) -> bool:
        return self.similarity(other) == 1

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


def num_nodes(flow): return len(flow.keys())
def num_edges(flow): return sum(len(v["wires"]) for v in flow.values())
def has_edge(flow, k1, k2): return k2 in flow[k1]["wires"]
def edge_to_string(k1, k2): return " --> ".join([k1, k2])
def string_to_edge(s): return tuple(s.split(" --> "))


def mapping_weight(node1, node2):
    # only makes sense to compare nodes of the same type
    # can add additional conditions here if needed
    try:
        mnode1 = {k: v for k, v in node1.items() if k != "wires"}
        mnode2 = {k: v for k, v in node2.items() if k != "wires"}
        obj1 = _Node(**mnode1)
        obj2 = _Node(**mnode2)
        ans = obj1.similarity(obj2)
    except Exception as e:
        print("Comparison Exception:", e)
        print(
            "comparing",
            json.dumps(node1, indent=2),
            "\nand\n",
            json.dumps(node2, indent=2),
        )
        ans = False
    return ans


def get_nodemap(flow1, flow2):
    nodemap = []
    for k1, v1 in flow1.items():
        for k2, v2 in flow2.items():
            wt = mapping_weight(v1, v2)
            if wt > 0:
                nodemap.append((k1, k2, wt))
    nodemap.sort(key=lambda x: (
        len(flow1[x[0]]["wires"]) + len(flow2[x[1]]["wires"])))
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

            if (not (e_a ^ e_b)) and (wta == 1.0) and (wtb == 1.0):
                # if e_a, e_b above have the same answer
                # AND the node mappings are perfect (all props match)
                # add the edge to the product graph
                ind1 = nmap.index((k1a, k2a, wta))
                ind2 = nmap.index((k1b, k2b, wtb))
                edge = (min(ind1, ind2), max(ind1, ind2))
                prodgraph.add(edge)

    return list(prodgraph)

def graph_corr(pgraph, nmap, flow1, flow2):
    G = nx.Graph()
    G.add_nodes_from(i + 1 for i in range(len(nmap)))
    G.add_edges_from([(a + 1, b + 1) for a, b in pgraph])
    clique = max(nx.algorithms.clique.find_cliques(G), key=lambda x: len(x))
    subset = [nmap[x - 1] for x in clique]
    if len(subset) > 1:
        for x in subset:
            assert x[2] == 1
    return subset


def find_correspondence(pgraph, nmap, flow1, flow2):
    if len(pgraph) == 0 and len(nmap) == 0:
        return [], True
    return graph_corr(pgraph, nmap, flow1, flow2)


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
                mapped_edges[edge_to_string(
                    k1a, k1b)] = edge_to_string(k2a, k2b)
    return mapped_edges

def node_similarity(subset, nodemap, flow1, flow2):
    if num_nodes(flow1) != 0 and num_nodes(flow2) != 0:
        yet_to_map = (
            set(flow1.keys()) - set(x[0] for x in subset),
            set(flow2.keys()) - set(x[1] for x in subset),
        )
        partial_map = dict()
        for k1, k2, wt in sorted(nodemap, key=lambda x: -x[2]):
            if k1 in yet_to_map[0] and k2 in yet_to_map[1]:
                partial_map[k1] = (k2, wt)
                yet_to_map[0].remove(k1)
                yet_to_map[1].remove(k2)

        frac_score = sum(x[1] for x in partial_map.values())

        score = len(subset) + frac_score
        answer = (score / num_nodes(flow1)) * (score / num_nodes(flow2))
        return answer
    else:
        return 0


def node_divergence(full1, full2, edges_only=True):
    flow1 = simplify_flow(full1)
    flow2 = simplify_flow(full2)
    nmap = get_nodemap(flow1, flow2)
    pg = create_product_graph(nmap, flow1, flow2)
    corr = find_correspondence(pg, nmap, flow1, flow2)

    return 1 - node_similarity(corr, nmap, flow1, flow2)

