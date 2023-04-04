#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time   : 2022/12/12 7:44
# @Author : 'Lou Zehua'
# @File   : tree_node.py

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import pandas as pd

from queue import Queue


class TreeNode(object):
    supported_trav_layer_info_tn_format = ["group_dict", "tuple_list"]
    pre_trav = []
    post_trav = []
    bfs_trav = []

    def __init__(self, val=None, parse=False):
        if parse:
            self.val = val.val
            self.parent = val.parent
            self.children = val.children
        else:
            self.val = val
            self.parent = None
            self.children = []

    def add_children(self, nodes):
        for node in nodes:
            self.add_child(node)
        return

    def add_child(self, node):
        if isinstance(node, TreeNode):
            self.children.append(node)
            node.parent = self
        else:
            raise TypeError(f"{node} is expected to be a instance of TreeNode, but got {type(node)}!")
        return

    @staticmethod
    def pre_traversal(tn):  # todo: add layer_info group_dict
        if not tn:
            TreeNode.pre_trav = []
            return TreeNode.pre_trav
        TreeNode.pre_trav.append(tn.val)
        for child_tn in tn.children:
            TreeNode.pre_traversal(child_tn)
        return TreeNode.pre_trav

    @staticmethod
    def post_traversal(tn):  # todo: add layer_info group_dict
        if not tn:
            TreeNode.post_trav = []
            return TreeNode.post_trav
        for child_tn in tn.children:
            TreeNode.post_traversal(child_tn)
        TreeNode.post_trav.append(tn.val)
        return TreeNode.post_trav

    @staticmethod
    def BFS(root_tn, layer_info=False, dtype_format="tuple_list", ret_elem_dtype="value"):
        if layer_info:
            supported_format = TreeNode.supported_trav_layer_info_tn_format
            if dtype_format not in supported_format:
                raise TypeError(f"TypeError: dtype_format only support these types within layer_info: {supported_format}")

        ret_elem_dtype_supported_types = ["TreeNode", "value"]
        if ret_elem_dtype not in ret_elem_dtype_supported_types:
            raise TypeError(f"TypeError: ret_elem_dtype must be in {ret_elem_dtype_supported_types}!")

        # init TreeNode.bfs_trav
        TreeNode.bfs_trav = []

        if not root_tn:
            if layer_info:
                TreeNode.bfs_trav = []  # default dtype_format == "tuple_list"
                if layer_info and dtype_format == "group_dict":
                    TreeNode.bfs_trav = {}
            return TreeNode.bfs_trav

        if not isinstance(root_tn, TreeNode):
            root_tn = TreeNode(root_tn, parse=True)

        # if root_tn.is_leaf:
        #     if not layer_info:
        #         TreeNode.bfs_trav = [root_tn]
        #     else:
        #         TreeNode.bfs_trav = [(0, root_tn)]  # default dtype_format == "tuple_list"
        #         if dtype_format == "group_dict":
        #             TreeNode.bfs_trav = TreeNode.tuple_list2group_dict(TreeNode.bfs_trav)
        #
        #     return TreeNode.bfs_trav

        Q = Queue()
        temp_layer_root_tn = (0, root_tn)
        Q.put(temp_layer_root_tn)

        visited_tns = []
        visited_tns.append(root_tn)

        visited_tn_infos = []
        temp_tn_info = root_tn if not layer_info else temp_layer_root_tn
        visited_tn_infos.append(temp_tn_info)

        visited_tnval_infos = []
        get_tuple_layer_treenodeval = lambda x: (x[0], x[1].val)
        temp_tnval_info = root_tn.val if not layer_info else get_tuple_layer_treenodeval(temp_layer_root_tn)
        visited_tnval_infos.append(temp_tnval_info)

        while not Q.empty():
            curr_layer, curr_tn = Q.get()
            # print(curr_tn.val, end=" ")
            for child_tn in curr_tn.children:
                if child_tn not in visited_tns:
                    if not isinstance(child_tn, TreeNode):
                        child_tn = TreeNode(child_tn, parse=True)

                    temp_layer_child_tn = (curr_layer + 1, child_tn)
                    Q.put(temp_layer_child_tn)

                    visited_tns.append(child_tn)
                    temp_tn_info = child_tn if not layer_info else temp_layer_child_tn
                    visited_tn_infos.append(temp_tn_info)
                    temp_tnval_info = child_tn.val if not layer_info else get_tuple_layer_treenodeval(temp_layer_child_tn)
                    visited_tnval_infos.append(temp_tnval_info)

        if layer_info:
            if ret_elem_dtype == "value":
                TreeNode.bfs_trav = visited_tnval_infos
            elif ret_elem_dtype == "TreeNode":
                TreeNode.bfs_trav = visited_tn_infos
            if dtype_format == "group_dict":
                TreeNode.bfs_trav = TreeNode.tuple_list2group_dict(TreeNode.bfs_trav)
        return TreeNode.bfs_trav

    @staticmethod
    def tuple_list2group_dict(trav):
        df_trav = pd.DataFrame(trav, columns=["layer", "value"])
        df_trav.set_index("value", inplace=True)
        return dict(df_trav.groupby("layer").groups)


if __name__ == '__main__':
    # A_________
    # |      \  \
    # B___   C  D_
    # | \ \     | \
    # E F G     H I
    root = TreeNode('A')
    B = TreeNode('B')
    root.add_child(B)
    root.add_child(TreeNode('C'))
    D = TreeNode('D')
    root.add_child(D)
    B.add_child(TreeNode('E'))
    B.add_child(TreeNode('F'))
    B.add_child(TreeNode('G'))
    D.add_child(TreeNode('H'))
    D.add_child(TreeNode('I'))

    pre_trav = TreeNode.pre_traversal(root)
    post_trav = TreeNode.post_traversal(root)
    print(pre_trav, post_trav)
    bfs_trav = TreeNode.BFS(root, layer_info=True, dtype_format="group_dict", ret_elem_dtype="TreeNode")
    print(bfs_trav)
    print(bfs_trav[2][3].parent.val)
