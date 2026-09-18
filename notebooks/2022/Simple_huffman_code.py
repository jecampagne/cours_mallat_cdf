# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: mallat
#     language: python
#     name: mallat
# ---

import numpy as np
from collections import Counter


# # Simple implementation of Huffman code

# +
class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return f'({self.left}, {self.right})'


def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()    
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        
    
    return nodes[0][0]

# -

string = 'blablablaaaaabcde'
freq = dict(Counter(string)) 
freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

# letter occurance
freq

node = make_tree(freq)

# get the code table
code_table = huffman_code_tree(node)

code_table

freq = dict(freq)

# compute Mean length and Entropy
R,H = np.sum([[freq[key]/len(string) * len(code_table[key]), -freq[key]/len(string) * np.log2(freq[key]/len(string))]  for key in freq], 
             axis=0)

print(f"Mean length of code: {R:.3},  Code entropy: {H:.3}, efficiency:{H/R:.3}")

# Perform the coding of the string
coded_string = ''.join([code_table[s] for s in string])

coded_string

# Reverse the code table and sort form smallest code (more frequent) to largest (less frequent)
decode_table = dict(sorted([(value, key) for (key, value) in code_table.items()], 
                      key=lambda x: len(x[0]), reverse=False))

decode_table


# simple decoding
def decode(s, decode_table):
    decoded_string = ''
    while len(s)>0:
        for c in decode_table :
            if s.startswith(c):
                decoded_string += decode_table[c]
                s= s[len(c):]
    return decoded_string


decode(coded_string, decode_table)


