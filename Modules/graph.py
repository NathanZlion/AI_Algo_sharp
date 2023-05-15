
from typing import Dict, List, Tuple


class Node:
    def __init__(self, name: str):
        self.name: str = name
        self._neighbors: Dict['Node', int|float] = {}


    def add_neighbor(self, neighbor: 'Node', weight:int|float = 0) -> None:
        """Adds the neighbor node to the node's neghbors list"""

        self._neighbors[neighbor] = weight


    def get_neighbors(self) -> Dict['Node', int|float]:
        """This returns a dictionary with key node and value int or float, \
            which represents the weight to that node"""

        return self._neighbors


    def get_weight(self, neighbor: 'Node'):
        """return the weight of edge between current node and neighbor."""

        return self._neighbors[neighbor] if neighbor in self._neighbors else float('inf')

    def number_of_edges(self):
        """returns the number of `neighbors (connections)` a node has."""

        return len(self._neighbors)
    
    def delete_neighbor(self, neighbor: 'Node') -> None:
        """delete a neighbor if it was indeed a neighbor."""

        if not neighbor in self._neighbors:
            return
     
        del self._neighbors[neighbor]


    def is_neighbor_of(self, node: 'Node') -> bool:
        """Returns if the neighbor has a connection with this node."""

        return node in self._neighbors


    def __str__(self) -> str:
        return self.name + " <neighbors> : "  + str([(neighbor.name, self.get_weight(neighbor)) for neighbor in self._neighbors.keys()])

class Graph:
    def __init__(self):
        self._nodes : Dict[str, Node]= {}


    def add_node(self, node: str) -> None:
        """Creates a new Node and adds it to the graph. It has no connections by\
            default."""
        if node in self._nodes: return
        self._nodes[node] = Node(node)


    def get_cost(self, node1: str, node2: str) -> int|float:
        """Returns the path cost between a and b"""
        return self._nodes[node1].get_weight(self._nodes[node2])


    def add_edge(self, node1: str, node2: str, weight:int|float=0) -> None:
        """Adds an edge between two nodes, and assigns a weight to the weight, \
            Weight is 0 by default."""

        if node1 not in self._nodes:
            self.add_node(node1)
        if node2 not in self._nodes:
            self.add_node(node2)
        self._nodes[node1].add_neighbor(self._nodes[node2], weight)
        self._nodes[node2].add_neighbor(self._nodes[node1], weight)


    def delete_node(self, node: str) -> None:
        """Deletes a node from theis graph if it exists. It also takes care of edges \
        formed with the node by deleting any edge made with it's neighbors."""

        if node in self._nodes:
            del self._nodes[node]
            for n in self._nodes:
                if node in self._nodes[n].get_neighbors():
                    del self._nodes[n].get_neighbors()[node]


    def delete_edge(self, node1: str, node2: str) -> None:
        """Deletes an edge between node1 and node2 if there exists one."""

        if node1 in self._nodes and node2 in self._nodes:
            self._nodes[node1].delete_neighbor(self._nodes[node2])
            self._nodes[node2].delete_neighbor(self._nodes[node1])


    def get_node(self, name: str) -> Node:
        """returns the node with the given name that is found in this grpah."""

        return self._nodes[name]


    def has_edge(self, node1: str, node2: str) -> bool:
        """returns True if there is an edge connecting node1 and node2 in this undirected graph."""

        return self._nodes[node1].is_neighbor_of(self._nodes[node2])
    

    def get_nodes(self) -> Dict[str, Node]:
        """returns a dictionary of all nodes in the current graph where the name of the node is the key \
            and the reference (pointer thing ...) to that node, the node it self, is the values."""

        return self._nodes


    def get_edges(self) -> List[Tuple[str, str, Dict[str, str|int|float]]]:
        """returns a list of edges between nodes that exist in this graph."""
        # ((node1, node2), `Weight`)
        weights : Dict[Tuple[str,str], int|float]= {}

        for node in self.get_nodes().values():
            for neighbor in node.get_neighbors():
                if not (node.name, neighbor.name) in weights and not (neighbor.name, node.name) in weights:
                    weights[(node.name, neighbor.name)] = node.get_weight(neighbor)

        return [(node1, node2, {'weight': weights[(node1, node2)]}) for (node1, node2) in weights]


    def get_inverted_edges(self):
        res = []
        for l in self.get_edges():
            res.append((l[0], l[1], {'weight': 1/l[2]['weight']}))  # type: ignore

        return res


    def get_adjecency_matrix(self) -> Tuple[Dict[str, int], List[List[int]]]:
        """returns the adjecency matrix of this graph."""

        # for every node assign an integer that is the row, col in the matrix
        n = self.get_number_of_nodes()
        map_node_index = {}
        ctr = 0
        for node in self._nodes:
            map_node_index[node] = ctr
            ctr +=1

        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]

        for edge in self.get_edges():
            node1, node2, weight = edge
            node1_index = map_node_index[node1]
            node2_index = map_node_index[node2]
            adj_matrix[node1_index][node2_index] = weight
            adj_matrix[node1_index][node1_index] = weight
        
        # return both dictionary and adj_matrix
        return (map_node_index, adj_matrix)

    def get_number_of_nodes(self) -> int:
        """Returns the number of nodes in the current graph.
        ```
        return self.__len__()
        ```
        """
        return self.__len__()

    def __len__(self) -> int:
        """Returns the number of nodes in this graph. you can also use the method `get_number_of_nodes` to do the same thing.
        ```
        # self._node is the mapping the has the nodes' names and reference.
        return len(self._nodes)
        ```
        """
        return len(self._nodes)


    def __str__(self) -> str:
        """prints graph."""
        lst = list()

        for node in self._nodes.values():
            lst.append(str(node))

        return str("\n".join(lst))

