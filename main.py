#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time   : 2022/12/10 22:06
# @Author : 'Lou Zehua'
# @File   : main.py 

import os
import sys

cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(cur_dir)  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)
print('Add root directory "{}" to system path.'.format(pkg_rootdir))


import re
import numpy as np
import pandas as pd
import textwrap


from script.tree_node import TreeNode


def data_preprocessing(df):
    # add your preprocessing function body here
    # filter
    open_source_license_valid = lambda x: str(x).lower().startswith('y')
    github_repo_link_valid = lambda x: pd.notna(x) and x != '-'
    df = df[(df["open_source_license"].apply(open_source_license_valid)) & (df["github_repo_link"].apply(github_repo_link_valid))]
    # format strs
    trim_open_source_license = lambda x: str(x).split('_')[0] if len(str(x)) else ''
    df.loc[:, "open_source_license"] = df.apply({"open_source_license": trim_open_source_license})
    return df


def get_kdata_vlabel_dict(df, kv_colnames):
    k_colname, v_colname = tuple(list(kv_colnames))
    return df.set_index(k_colname)[v_colname].to_dict()


def get_klabel_vdatalist_dict(df, kv_colnames, groupby_idx=-1):
    kv_colnames = list(kv_colnames)
    groupby_colname = kv_colnames[groupby_idx]
    k_colname, v_colname = tuple(kv_colnames[:2])
    df.set_index(k_colname, inplace=True)
    return dict(df.groupby(groupby_colname).groups)


def format_on_keys(s: str, d: dict, strict=False):
    if strict:
        return s.format(**d)

    s = re.sub(r"{(\w+)}", r"{{\1}}", s)
    for k in d.keys():
        s = s.replace("{{" + k + "}}", "{" + k + "}")
    return s.format(**d)


def rebuild_samevars_in_str(s, varname, sep="__", start_idx=0):
    s = str(s)
    varname = str(varname)
    sep = str(sep)
    varname_pat = "{" + str(varname) + "}"
    s_parts = s.split(varname_pat)
    s_rebuild = ''
    for i in range(len(s_parts)):
        s_rebuild += s_parts[i]
        if i <= len(s_parts) - 2:
            s_rebuild += "{" + varname + sep + str(start_idx + i) + "}"
    return s_rebuild


def gen_issue_body_format_str(data_dict, level_pattern_dict, level_start=0):
    level_pattern_kasc_dict = dict(sorted(level_pattern_dict.items(), key=lambda x: x[0], reverse=False))
    pat_vars_list = []
    for pat in level_pattern_kasc_dict.values():
        pat_vars = re.findall(r"\{(\w+)\}", pat)
        pat_vars_list.append(pat_vars)

    dtypes = ['dict', 'list', 'final']
    valid_parentalias = lambda x: isinstance(x, str)
    valid_level = lambda x: str(x).isdigit()
    valid_dtype = lambda x: x in dtypes
    valid_attrrole = lambda x, parent_dtype: {"dict": x in ['keys', 'values'], "list": x == 'elements', "final": x == 'forest_virtual_root'}[parent_dtype]
    valid_childdtype = lambda x: x in dtypes
    valid_childalias = lambda x: isinstance(x, str)
    pattern_value_error = "ValueError: The pattern value format: [parentalias]__level_dtype:{dict|list|final}_attrrole:{keys|values|elements}[__childdtype:{dict|list|final}][__childalias]"
    FOREST_ROOT_LEVEL_START = 0
    OFFSET = FOREST_ROOT_LEVEL_START + 1 - level_start
    assert(FOREST_ROOT_LEVEL_START + 1 == level_start + OFFSET)  # 新增的forest_virtual_root占用了level 0
    forest_root_node_value = {
        "parent_node_info": {"parentalias": ""},
        "node_body": {"level": FOREST_ROOT_LEVEL_START, "dtype": "str", "attrrole": "forest_virtual_root"},
        "child_node_info": {"childdtype": "list", "childalias": ""},
        "pat_var": ""
    }
    forest_virtual_root = TreeNode(forest_root_node_value)
    for pat_vars in pat_vars_list:
        for pat_var in pat_vars:
            # [parentalias]__level_dtype:{dict|list|final}_attrrole:{keys|values|elements}[__childdtype:{dict|list|final}][__childalias]
            # 即 [parentalias]__level_dtype_attrrole[__childdtype][__childalias]
            pat_var_params = str(pat_var).split("__")
            parentalias = pat_var_params[0]
            if not valid_parentalias(parentalias):
                raise ValueError(pattern_value_error)
            level, dtype, attrrole = pat_var_params[1].split('_')[:3]
            norm_dtype = re.sub('\d', '', dtype)
            if not valid_level(level) and valid_dtype(norm_dtype) and valid_attrrole(attrrole, norm_dtype):
                raise ValueError(pattern_value_error)

            childdtype = ''
            childalias = ''
            if len(pat_var_params) >= 3:
                childdtype = pat_var_params[2]
                norm_childdtype = re.sub('\d', '', childdtype)
                if not valid_childdtype(norm_childdtype):
                    raise ValueError(pattern_value_error)
                if len(pat_var_params) >= 4:
                    childalias = pat_var_params[3]
                    if not valid_childalias(childalias):
                        raise ValueError(pattern_value_error)
            # print(parentalias)
            # print(level, dtype, attrrole)
            # print(childdtype, childalias)
            # print('---')

            node_value = {
                "parent_node_info": {"parentalias": parentalias},
                "node_body": {"level": level, "dtype": dtype, "attrrole": attrrole},
                "child_node_info": {"childdtype": childdtype, "childalias": childalias},
                "pat_var": pat_var
            }
            tn = TreeNode(node_value)

            try:
                curr_layer = int(tn.val["node_body"]["level"]) + OFFSET
            except TypeError:
                raise TypeError("TypeError: level settings must be a integer number!")

            if curr_layer <= 0:
                raise ValueError("ValueError: level settings must be a integer number!")
            elif curr_layer == 1:
                forest_virtual_root.val["child_node_info"]["childdtype"] = tn.val["node_body"]["dtype"]
                forest_virtual_root.val["child_node_info"]["childalias"] = tn.val["parent_node_info"]["parentalias"]
                parent_layer_tns = [forest_virtual_root]
            else:
                # TreeNode.BFS(forest_virtual_root, layer_info=True, dtype_format="group_dict", ret_elem_dtype="TreeNode")
                layer_tn_group_dict = TreeNode.bfs_trav
                parent_layer = curr_layer - 1
                parent_layer_tns = layer_tn_group_dict[parent_layer]

            parent_tn = None
            for parent_layer_tn in parent_layer_tns:
                if parent_layer_tn.val["child_node_info"]["childalias"] == tn.val["parent_node_info"]["parentalias"]:
                    parent_tn = parent_layer_tn
            if not parent_tn:
                print(f"Warning: the parent_tn of pat_var is not exist, please check 'parentalias'! Try to ignore unexpected pat_var: {pat_var}!")
            parent_tn.add_child(tn)
            TreeNode.BFS(forest_virtual_root, layer_info=True, dtype_format="group_dict", ret_elem_dtype="TreeNode")

    # Format string
    # bfs_trav_groupby_layer_desc = sorted(TreeNode.bfs_trav.items(), key=lambda x: x[0], reverse=True)  # 按layer逆序format
    bfs_trav_groupby_layer_asc = dict(sorted(TreeNode.bfs_trav.items(), key=lambda x: x[0]))  # 按layer逆序format

    def gen_curr_layer_format_str(curr_layer, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, dataset_handler_layers_dicts, offset=OFFSET):
        if curr_layer == 0:  # 跳过额外增加的虚拟根节点
            return gen_curr_layer_format_str(curr_layer + 1, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, dataset_handler_layers_dicts, offset=OFFSET)

        content_pattern_formated = ''

        tns = bfs_trav_groupby_layer_asc[curr_layer]
        recovered_level = curr_layer - offset
        content_pattern_curr_layer = str(level_pattern_kasc_dict[recovered_level])

        dataset = dataset_handler_layers_dicts[recovered_level]

        curr_layer_pat_dict = {}

        curr_layer_formatable = []

        # reflect: 优先递归解决无法format的
        def sortedby_key_reflect(list_k, list_v, asc=True):
            list_k = list(list_k)
            list_v = list(list_v)
            k_v_pairs = list(zip(list_k, list_v))
            k_v_pairs = list(sorted(k_v_pairs, key=lambda x: x[0], reverse=not asc))
            return list(zip(*k_v_pairs))[1]

        dataset_handler_currlayer_dicts = {}
        for tn in tns:
            pat_var = tn.val["pat_var"]
            if tn.val["node_body"]["dtype"].startswith('dict'):
                dataset = dict(dataset)
                func_name = tn.val["node_body"]["attrrole"]
                temp_data_list = getattr(dataset, func_name)()
            elif tn.val["node_body"]["dtype"].startswith('list'):
                temp_data_list = list(dataset[tn.parent.val["pat_var"]])
                assert (tn.val["node_body"]["attrrole"] == "elements")
            else:
                raise ValueError("ValueError: The pattern value format: [parentalias]__level_dtype:{dict|list|final}_attrrole:{keys|values|elements}[__childdtype:{dict|list|final}][__childalias]")
            dataset_handler_currlayer_dicts[pat_var] = temp_data_list
            if tn.val["child_node_info"]["childdtype"] == "final" or not len(tn.children):
                temp_formatable = True
            else:
                temp_formatable = False
            curr_layer_formatable.append(temp_formatable)
        dataset_handler_layers_dicts[curr_layer] = dataset_handler_currlayer_dicts

        idx_sortedby_formatable_asc = sortedby_key_reflect(curr_layer_formatable, list(range(len(curr_layer_formatable))))
        if all(curr_layer_formatable):
            for tn in tns:
                pat_var = tn.val["pat_var"]
                if tn.val["node_body"]["dtype"].startswith('dict'):
                    dataset = dict(dataset)
                    func_name = tn.val["node_body"]["attrrole"]
                    temp_data_list = getattr(dataset, func_name)()
                    # content_pattern_formated += content_pattern_curr_layer.format(**curr_layer_pat_dict)
                    for elem in temp_data_list:
                        curr_layer_pat_dict[pat_var] = elem
                        content_pattern_formated += format_on_keys(content_pattern_curr_layer, curr_layer_pat_dict, strict=False)
                elif tn.val["node_body"]["dtype"].startswith('list'):
                    temp_data_list = list(dataset[tn.parent.val["pat_var"]])
                    assert(tn.val["node_body"]["attrrole"] == "elements")
                    for elem in temp_data_list:
                        curr_layer_pat_dict[pat_var] = elem
                        content_pattern_formated += format_on_keys(content_pattern_curr_layer, curr_layer_pat_dict, strict=False)
                tn.val["child_node_info"]["childdtype"] = "final"  # reset final after get formated str
            return content_pattern_formated
        else:
            dataset_handler_currlayer_dicts = dataset_handler_layers_dicts[curr_layer]
            assert(len(idx_sortedby_formatable_asc) == len(dataset_handler_currlayer_dicts))
            for i in idx_sortedby_formatable_asc:
                temp_keys = list(dataset_handler_currlayer_dicts.keys())
                dataset_handler_unformatable_values = list(dataset_handler_currlayer_dicts[temp_keys[i]])
                temp_tns = bfs_trav_groupby_layer_asc[curr_layer]
                temp_tn = temp_tns[i]
                if not curr_layer_formatable[i]:
                    if temp_tn.val["node_body"]["dtype"].startswith("dict"):
                        for j in range(len(dataset_handler_unformatable_values)):
                            temp_dataset_handler_layers_dicts = {
                                curr_layer: {temp_keys[i]: dataset_handler_unformatable_values[j]}}
                            # dataset_handler_layers_dicts[curr_layer][temp_keys[i]] = dataset_handler_unformatable_values[j]
                            dataset_handler_unformatable_values[j] = gen_curr_layer_format_str(curr_layer + 1,
                                                                                          bfs_trav_groupby_layer_asc,
                                                                                          level_pattern_kasc_dict,
                                                                                          temp_dataset_handler_layers_dicts,
                                                                                          offset=OFFSET)
                    elif temp_tn.val["node_body"]["dtype"].startswith("list"):
                        for j in range(len(dataset_handler_unformatable_values)):
                            temp_dataset_handler_layers_dicts = {curr_layer: {temp_keys[i]: dataset_handler_unformatable_values[j]}}
                            dataset_handler_unformatable_values[j] = gen_curr_layer_format_str(curr_layer + 1, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, temp_dataset_handler_layers_dicts, offset=OFFSET)
                    dataset_handler_currlayer_dicts[temp_keys[i]] = dataset_handler_unformatable_values
                else:
                    dataset_handler_currlayer_dicts[temp_keys[i]] = list(dataset_handler_currlayer_dicts[temp_keys[i]])
                temp_tn.val["child_node_info"]["childdtype"] = "final"  # reset final after get formated str
            dataset_currlayer_record_list = pd.DataFrame(dataset_handler_currlayer_dicts).to_dict("records")
            for record_dict in dataset_currlayer_record_list:
                content_pattern_formated += format_on_keys(content_pattern_curr_layer, record_dict, strict=False)
            return content_pattern_formated

    curr_layer = 0
    content_pattern_formated = gen_curr_layer_format_str(curr_layer, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, {curr_layer: data_dict}, offset=OFFSET)
    return content_pattern_formated


def gen_issue_body_format_str_simple(label_datalist_dict):
    content_pattern_formated = ''
    for k, v in label_datalist_dict.items():
        value_pattern_formated = ''
        for v_elem in v:
            value_pattern_formated += value_pattern.format(each_repo_pat__1_list_elements__final=v_elem)
        content_pattern_formated += content_pattern.format(__0_dict0_keys__final=k, __0_dict0_values__list__each_repo_pat=value_pattern_formated)
    return content_pattern_formated


if __name__ == '__main__':
    labeled_data_path = './data/DB_EngRank_full_202212.csv'
    df = pd.read_csv(labeled_data_path)
    df = data_preprocessing(df)
    kv_colnames = ["github_repo_link", "category_label"]
    # data_label_pairs = get_kdata_vlabel_dict(df, kv_colnames)
    label_datalist_dict = get_klabel_vdatalist_dict(df, kv_colnames)
    # print(label_datalist_dict)

    # 一个pattern内只能包含同级的信息，两级之间使用"__"分隔，每一级的类型只能是dict, list, str，变量命令格式为：
    # [parentalias]__level_dtype:{dict|list|final}_attrrole:{keys|values|elements}[__childdtype:{dict|list|final}][__childalias]
    # 即 [parentalias]__level_dtype_attrrole[__childdtype][__childalias]
    # 当下一级有子pattern时，需要给出dtype信息，若不给出则默认到了叶子结点。
    content_pattern = '''\
    Label: {__0_dict0_keys__final}
    
    Type: Tech-1
    
    Repos:
    
    {__0_dict0_values__list__each_repo_pat}
    '''
    content_pattern = textwrap.dedent(content_pattern)

    value_pattern = '''\
    - {each_repo_pat__1_list_elements__final}
    '''
    value_pattern = textwrap.dedent(value_pattern)

    level_pattern_dict = {
        0: content_pattern,
        1: value_pattern
    }
    issue_body_str = gen_issue_body_format_str(label_datalist_dict, level_pattern_dict)
    # print(issue_boddy_str)

    path_issue_body_format_txt = './data/issue_body_format.txt'
    with open(path_issue_body_format_txt, 'w') as f:
        f.write(issue_body_str)
