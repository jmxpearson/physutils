"""
Implement a simple version of the union-find data structure.
Uses a list of tuples, which is probably slow, but accommodates a 
flexible number of nodes.
Does not implement path compression on lookup.
"""
import unittest

class UnionFind(object):
    def __init__(self):
        self.nodes = {}

    def add(self, item):
        if item not in self.nodes:
            # if item not already in nodes, add a tuple:
            # (parent, number of nodes attached below)
            self.nodes[item] = (item, 1)

    def find(self, item):
        if item in self.nodes:
            parent, weight = self.nodes[item]
            if parent == item:
                return (parent, weight)
            else:
                return self.find(parent)

    def union(self, x, y):
        rx, wx = self.find(x)
        ry, wy = self.find(y)

        if rx == ry:
            return

        # join smaller tree to root of larger tree
        # in case of ties, join root(y) under root(x)
        if wx < wy:
            self.nodes[rx] = (ry, wx)
            self.nodes[ry] = (ry, wy + wx)
        else:
            self.nodes[ry] = (rx, wy)
            self.nodes[rx] = (rx, wx + wy)

class UnionFindTests(unittest.TestCase):
    def setUp(self):
        self.uf = UnionFind()

    def test_constructor_returns_empty_dict(self):
        self.assertEqual(self.uf.nodes, {})

    def test_find_on_empty_uf_returns_none(self):
        self.assertIsNone(self.uf.find(1))

    def test_can_add_item(self):
        self.uf.add(5)
        self.assertEqual(self.uf.find(5), (5, 1))

        self.uf.add((4,4))
        self.assertEqual(self.uf.find((4,4)), ((4,4), 1))

    def test_can_union_items(self):
        self.uf.add(1)
        self.uf.add(5)

        self.uf.union(1, 5)

        # check that one node has become parent of the other
        self.assertTrue(self.uf.find(1)[0] == 5 or self.uf.find(5)[0] == 1)

    def test_tied_union_makes_first_argument_root(self):
        self.uf.add('a')
        self.uf.add('b')

        self.uf.union('a', 'b')
        self.assertEqual(self.uf.find('a'), ('a', 2))
        self.assertEqual(self.uf.find('b'), ('a', 2))

    def test_union_makes_largest_tree_root(self):
        self.uf.add('a')
        self.uf.add('b')
        self.uf.add('c')
        self.uf.add('d')

        self.uf.union('a', 'b')
        self.uf.union('c', 'd')
        self.uf.union('a', 'd')

        self.uf.add('e')
        self.uf.add('f')
        self.uf.add('g')

        self.uf.union('e', 'f')
        self.uf.union('e', 'g')

        self.uf.union('e', 'a')

        self.assertEqual(self.uf.find('g')[0], 'a')

    def test_union_correctly_sums_weights(self):
        self.uf.add('a')
        self.uf.add('b')
        self.uf.add('c')
        self.uf.add('d')

        self.uf.union('a', 'b')
        self.uf.union('c', 'd')
        self.uf.union('a', 'd')

        self.uf.add('e')
        self.uf.add('f')
        self.uf.add('g')

        self.uf.union('e', 'f')
        self.uf.union('e', 'g')

        self.assertEqual(self.uf.find('g')[1], 3)
        self.assertEqual(self.uf.find('a')[1], 4)

        self.uf.union('a', 'g')
        self.assertEqual(self.uf.find('a')[1], 7)

    def test_find_recurses_all_the_way_to_root(self):
        self.uf.add('a')
        self.uf.add('b')
        self.uf.add('c')
        self.uf.add('d')

        self.uf.union('a', 'b')
        self.uf.union('c', 'd')
        self.uf.union('a', 'd')

        self.assertEqual(self.uf.find('d')[0], 'a')

    def test_union_already_connected_elements_preserves_weight(self):
        self.uf.add('a')
        self.uf.add('b')
        self.uf.add('c')
        self.uf.add('d')

        self.uf.union('a', 'b')
        self.uf.union('c', 'd')
        self.uf.union('a', 'd')

        w1 = self.uf.find('a')[1]
        self.uf.union('d', 'b')
        w2 = self.uf.find('a')[1] 
        self.assertEqual(w1, w2)


if __name__ == '__main__':
    unittest.main()