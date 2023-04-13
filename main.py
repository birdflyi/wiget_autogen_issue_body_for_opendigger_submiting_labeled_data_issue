#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.7

# @Time   : 2022/12/10 22:06
# @Author : 'Lou Zehua'
# @File   : main.py 

import os
import sys

if '__file__' not in globals():
    # !pip install ipynbname  # Remove comment symbols to solve the ModuleNotFoundError
    import ipynbname

    nb_path = ipynbname.path()
    __file__ = str(nb_path)
cur_dir = os.path.dirname(__file__)
pkg_rootdir = cur_dir  # os.path.dirname()向上一级，注意要对应工程root路径
if pkg_rootdir not in sys.path:  # 解决ipynb引用上层路径中的模块时的ModuleNotFoundError问题
    sys.path.append(pkg_rootdir)
    print('-- Add root directory "{}" to system path.'.format(pkg_rootdir))

import re
import shutil
import textwrap

import pandas as pd

from script.tree_node import TreeNode


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


def data_preprocessing(df, filter_has_github_repo_link=True, filter_with_open_source_license=False):
    # add your preprocessing function body here
    # filter
    open_source_license_valid = lambda x: str(x).lower().startswith('y') if filter_with_open_source_license else True  # always return True to ignore filter
    github_repo_link_valid = lambda x: pd.notna(x) and x != '-' if filter_has_github_repo_link else True
    df = df[(df["open_source_license"].apply(open_source_license_valid)) & (df["github_repo_link"].apply(github_repo_link_valid))].copy()
    # format strs
    trim_open_source_license = lambda x: str(x).split('_')[0] if len(str(x)) else ''
    df.loc[:, "open_source_license"] = df.apply({"open_source_license": trim_open_source_license})
    return df


def get_klabel_vdatalist_dict(df, kv_colnames, multi_onehot_label_cols=True, df_dedup_base=None, base_filter=None):
    kv_colnames = list(kv_colnames)
    group_dict = {}
    if not multi_onehot_label_cols:
        k_colname, v_colname = kv_colnames[:2]
        groupby_colname = v_colname
        if df_dedup_base is not None:
            df_dedup_base = pd.DataFrame(df_dedup_base)
            df = pd.concat([df, df_dedup_base, df_dedup_base], axis=0).drop_duplicates(subset=[k_colname], keep=False)
        df.set_index(k_colname, inplace=True)
        group_dict = dict(df.groupby(groupby_colname).groups)
    else:
        k_colname, v_colnames = kv_colnames[:2]
        groupby_colnames = list(v_colnames)
        groupby_idxes = list(range(len(groupby_colnames)))
        if df_dedup_base is not None:
            df_dedup_base = pd.DataFrame(df_dedup_base)
            df_dedup_base.set_index(k_colname, inplace=True)
        df.set_index(k_colname, inplace=True)
        for each_groupby_idx in groupby_idxes:
            groupby_colname = groupby_colnames[each_groupby_idx]
            filter = df[groupby_colname] != 0
            if df_dedup_base is not None and groupby_colname is not None:
                filter = filter & (filter ^ base_filter(df_dedup_base, groupby_colname))  # use xor to deal with records not in df_dedup_base
            group_dict[groupby_colname] = list(df[filter].index)
    return group_dict


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


def gen_curr_layer_format_str(curr_layer, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, dataset_handler_layers_dicts, offset):
    if curr_layer == 0:  # 跳过额外增加的虚拟根节点
        return gen_curr_layer_format_str(curr_layer + 1, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, dataset_handler_layers_dicts, offset)

    content_pattern_formated = ''

    tns = bfs_trav_groupby_layer_asc[curr_layer]
    recovered_level = curr_layer - offset
    content_pattern_curr_layer = str(level_pattern_kasc_dict[recovered_level])

    dataset = dataset_handler_layers_dicts[recovered_level]
    if dataset_handler_layers_dicts.get("del_emptyval_data"):
        temp_dataset = {}
        for k, v in dict(dataset).items():
            if len(v):
                temp_dataset[k] = v
        dataset = temp_dataset

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
                                                                                      offset)
                elif temp_tn.val["node_body"]["dtype"].startswith("list"):
                    for j in range(len(dataset_handler_unformatable_values)):
                        temp_dataset_handler_layers_dicts = {curr_layer: {temp_keys[i]: dataset_handler_unformatable_values[j]}}
                        dataset_handler_unformatable_values[j] = gen_curr_layer_format_str(curr_layer + 1, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, temp_dataset_handler_layers_dicts, offset)
                dataset_handler_currlayer_dicts[temp_keys[i]] = dataset_handler_unformatable_values
            else:
                dataset_handler_currlayer_dicts[temp_keys[i]] = list(dataset_handler_currlayer_dicts[temp_keys[i]])
            temp_tn.val["child_node_info"]["childdtype"] = "final"  # reset final after get formated str
        dataset_currlayer_record_list = pd.DataFrame(dataset_handler_currlayer_dicts).to_dict("records")
        for record_dict in dataset_currlayer_record_list:
            content_pattern_formated += format_on_keys(content_pattern_curr_layer, record_dict, strict=False)
        return content_pattern_formated


def gen_issue_body_format_str(data_dict, level_pattern_dict, level_start=0, del_emptyval_data=True):
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

    curr_layer = 0
    content_pattern_formated = gen_curr_layer_format_str(curr_layer, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict, {curr_layer: data_dict, "del_emptyval_data": del_emptyval_data}, offset=OFFSET)
    return content_pattern_formated


def gen_issue_body_format_str_simple(label_datalist_dict):
    content_pattern_formated = ''
    for k, v in label_datalist_dict.items():
        value_pattern_formated = ''
        for v_elem in v:
            value_pattern_formated += value_pattern.format(each_repo_pat__1_list_elements__final=v_elem)
        content_pattern_formated += content_pattern.format(__0_dict0_keys__final=k, __0_dict0_values__list__each_repo_pat=value_pattern_formated)
    return content_pattern_formated


def to_yaml_parts(src_path, tar_dir=None, encoding='utf-8', pre_format_conf=None, filename_reg=None, sep="===="):
    if not tar_dir:
        tar_dir = os.path.dirname(src_path)
    if not os.path.isdir(tar_dir):
        os.makedirs(tar_dir)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"FileNotFoundError: [Errno 2] No such file or directory: {src_path}. You may skip "
              f"the step 2. Create data issue in open-digger: save content generated with `/parse-github-id` option as 'issue_body_format_parse_github_id.txt'")
    with open(src_path, 'r', encoding=encoding) as f:
        s = f.read()
        if pre_format_conf:
            for reg, repl in dict(pre_format_conf).items():
                s = re.sub(r'{}'.format(reg), '{}'.format(repl), s)
        yaml_formatstrs = s.split(sep=sep)
        yaml_formatstrs = [x.strip() for x in yaml_formatstrs if x.strip() != '']
        yaml_filename_formatstr_dict = {}
        suffix = '.yml'
        for i in range(len(yaml_formatstrs)):
            yaml_formatstr = yaml_formatstrs[i]
            if filename_reg:
                curr_yaml_filename = re.findall(r'{}'.format(filename_reg), yaml_formatstr)[0]
                curr_yaml_filename = str(curr_yaml_filename).strip().lower()
                curr_yaml_filename = re.sub(r"[ -]", "_", curr_yaml_filename)
            else:
                curr_yaml_filename = str(i)
            curr_yaml_filename_suffix = curr_yaml_filename + suffix
            yaml_filename_formatstr_dict[curr_yaml_filename_suffix] = yaml_formatstr
        for fname, formatstr in yaml_filename_formatstr_dict.items():
            curr_yaml_path = os.path.join(tar_dir, fname)
            with open(curr_yaml_path, 'w', encoding=encoding) as f:
                f.write(formatstr)
                f.write("\n")
            print(f"{curr_yaml_path} is saved!")
    return


# ------------------------1. Auto generate [data] issue body for opendigger-------------------
def auto_gen_issue_body_for_opendigger(df, issue_body_format_txt_path, use_column__mode_col_config=(2, None),
                                       del_emptyval_data=True, incremental=False, **kwargs):
    df = data_preprocessing(df, filter_has_github_repo_link=kwargs.get("filter_has_github_repo_link", True),
                            filter_with_open_source_license=kwargs.get("filter_with_open_source_license", False))
    df = pd.DataFrame(df)

    use_column_config_modes = ["category_label_col", "multimodel_labels_col", "multimodel_onehot_label_cols"]
    use_column_config_mode, use_column_config_col = tuple(use_column__mode_col_config)[:2]

    if type(use_column_config_mode) is int:
        idx = use_column_config_mode
        use_column_config_mode = use_column_config_modes[idx]
    print(f"use_column_config_mode: {use_column_config_mode}. All supported settings(you can also use index): {use_column_config_modes}.")
    assert (use_column_config_mode in use_column_config_modes)

    msg_use_column_config_setting_error = f"use_column_config_mode must be in {use_column_config_modes}, while {use_column_config_mode} is got!"
    if use_column_config_mode == "category_label_col":
        if use_column_config_col:
            if not isinstance(use_column_config_col, str):
                raise TypeError(f"Mode:{use_column_config_mode}. use_column_config_col must be a str or None!")

        use_column_config_col = use_column_config_col or "category_label"
        kv_colnames = ["github_repo_link", use_column_config_col]
        multi_onehot_label_cols = False

    elif use_column_config_mode.lower().startswith("multimodel"):
        default_onehot_label_cols = ["Content", "Document", "Event", "Graph", "Key-value", "Multivalue", "Native XML",
                                     "Navigational", "Object oriented", "RDF", "Relational", "Search engine",
                                     "Spatial DBMS",
                                     "Time Series", "Wide column"]

        if use_column_config_mode == "multimodel_labels_col":
            if use_column_config_col:
                if not isinstance(use_column_config_col, str):
                    raise TypeError(f"Mode:{use_column_config_mode}. use_column_config_col must be a str or None!")

            # 将Database Model, Multi_model_info两列中的str按','切分为类型列表，nan则返回[]，最后series纵向求和，得到拼接列表，去重后得到标签列表
            use_column_config_col = use_column_config_col or "Multi_model_info"
            series_Multi_model_info = df[use_column_config_col]
            func_str_split_nan_as_emptylist = lambda x, seps: [s.strip() for s in re.split('|'.join(seps), str(x))] if pd.notna(x) else []
            substr_list_Multi_model_info = series_Multi_model_info.apply(func_str_split_nan_as_emptylist, seps=[',', '#dbdbio>\|<dbengines#'])
            types_Multi_model_info = list(set(substr_list_Multi_model_info.sum()))
            types_Multi_model_info.sort()
            if len(set(df.columns) & set(types_Multi_model_info)) > 0:
                raise ValueError(f"The values in column {use_column_config_col} conflict with column names in df.columns!")

            temp_d = {}
            for k_type in types_Multi_model_info:
                temp_k_type_onehot_vec = {}
                for i in pd.Series(substr_list_Multi_model_info).index:
                    onehot = 1 if k_type in substr_list_Multi_model_info[i] else 0
                    temp_k_type_onehot_vec[i] = onehot
                temp_d[k_type] = temp_k_type_onehot_vec
            temp_df = pd.DataFrame().from_dict(temp_d)
            assert(all(df.index == temp_df.index))
            df = pd.concat([df, temp_df], axis=1)
            onehot_label_cols = types_Multi_model_info

        elif use_column_config_mode == "multimodel_onehot_label_cols":
            if use_column_config_col:
                if not isinstance(use_column_config_col, list):
                    raise TypeError(f"Mode:{use_column_config_mode}. use_column_config_col must be a list or None!")
            onehot_label_cols = use_column_config_col or default_onehot_label_cols
        else:
            raise ValueError(msg_use_column_config_setting_error)
        # use multi-model onehot_label_cols
        onehot_label_cols = onehot_label_cols or default_onehot_label_cols
        kv_colnames = ["github_repo_link", onehot_label_cols]
        multi_onehot_label_cols = True
    else:
        raise ValueError(msg_use_column_config_setting_error)

    df_last_v = None
    base_filter = None
    if incremental:
        last_v_labeled_data_path = kwargs.get("last_v_labeled_data_path")
        if not last_v_labeled_data_path:
            print("ParamError: incremental mode must specify last_v_labeled_data_path for the last version of data.")
            return
        df_last_v = pd.read_csv(last_v_labeled_data_path)
        df_last_v = data_preprocessing(df_last_v, filter_has_github_repo_link=kwargs.get("filter_has_github_repo_link", True),
                            filter_with_open_source_license=kwargs.get("filter_with_open_source_license", False))
        # Why [df, df_last_v, df_last_v]:
        # 1. df contains the newest data, drop_duplicates will keep the first hit record.
        # 2. [df_last_v, df_last_v] duplicates every records in df_last_v, which will be dropped by drop_duplicates.
        df = pd.concat([df, df_last_v, df_last_v], axis=0).drop_duplicates(subset=[kv_colnames[0], 'category_label'], keep=False)
        base_filter = lambda df, label_str: df["category_label"].apply(lambda x: label_str in x if pd.notna(x) else False)
    label_datalist_dict = get_klabel_vdatalist_dict(df, kv_colnames, multi_onehot_label_cols, df_dedup_base=df_last_v, base_filter=base_filter)
    # print(label_datalist_dict)

    issue_body_str = gen_issue_body_format_str(label_datalist_dict, level_pattern_dict, del_emptyval_data=del_emptyval_data)
    # print(issue_boddy_str)

    with open(issue_body_format_txt_path, 'w') as f:
        f.write(issue_body_str)


def df_getRepoId_to_yaml(src_path, tar_dir=None):
    if not tar_dir:
        tar_dir = os.path.dirname(src_path)

    sep = "#====#"
    pre_format_conf_kpattern_vstdreplacement = {
        "\n- (.*)": "\n    - \\1",
        "\n?Label: ([^\n]+)\n": f"\n{sep}name: Database - \\1\n",
        "\n+": "\n",
        "Type: Tech-1": "type: Tech-1",
        "Repos:\n": "data:\n  github_repo:\n",
        "name: Database - Object oriented": "name: Database - Object Oriented",
        "name: Database - Search engine": "name: Database - Search Engine",
        "name: Database - Spatial DBMS": "name: Database - Spatial",
        "name: Database - Wide column": "name: Database - Wide Column",

    }
    filename_reg = "name: Database - ([^\n]+)\n"
    to_yaml_parts(src_path=src_path, tar_dir=tar_dir, pre_format_conf=pre_format_conf_kpattern_vstdreplacement, filename_reg=filename_reg, sep=sep)
    return


def get_filenames_from_dir(dir_str, suffix='.yml', recursive=False):
    suffix_filter = lambda fname: fname.lower().endswith(suffix) if suffix is not None else True
    recursive_filter = lambda subdir: True if recursive else not len(directories)
    filenames = []
    for root, directories, files in os.walk(dir_str):
        for filename in files:
            if not recursive_filter or not suffix_filter(filename):
                continue
            filenames.append(filename)
    return filenames


def merge_data_incremental_order(last_v_path, inc_path, tar_path, encoding='utf-8'):
    shutil.copyfile(last_v_path, tar_path)
    with open(inc_path, 'r', encoding=encoding) as f:
        yaml_formatstr = f.read()
    github_repo_text_header = "\n  github_repo:"
    github_repo_text_paragraph_pattern = r"(\n  github_repo:((\n    - [^\n]*)*))"
    yaml_data_github_repo_formatstr_matchlist = re.findall(github_repo_text_paragraph_pattern, yaml_formatstr)[0]
    with open(tar_path, 'r+', encoding=encoding) as f:
        tar_text = f.read().strip('\n')
        github_repo_text_header_is_exist = tar_text.find(github_repo_text_header) >= 0
        yaml_data_github_repo_formatstr = yaml_data_github_repo_formatstr_matchlist[int(github_repo_text_header_is_exist)]
        tar_text += yaml_data_github_repo_formatstr
        f.seek(0, 0)
        f.write(tar_text)
        print(f'Warning: {tar_path} is updated! Manual check may be required!!!')


def auto_gen_current_version_incremental_order_merged(last_v_dir, inc_dir, curr_merged_tar_dir, suffix='.yml'):
    last_v_filenames = get_filenames_from_dir(last_v_dir, suffix=suffix, recursive=False)
    inc_filenames = get_filenames_from_dir(inc_dir, suffix=suffix, recursive=False)
    curr_merged_filenames = last_v_filenames
    UNDEFINED = 0  # 00000000
    ONLY_LAST_V_EXIST = 1  # 00000001
    ONLY_INC_EXIST = 2  # 00000010
    LAST_V_INC_BOTH_EXIST = ONLY_LAST_V_EXIST | ONLY_INC_EXIST  # 00000011
    curr_merged_states = [ONLY_LAST_V_EXIST] * len(curr_merged_filenames)
    curr_merged_filenames_states_dict = dict(zip(curr_merged_filenames, curr_merged_states))
    for filename in inc_filenames:
        curr_merged_filenames_states_dict[filename] = curr_merged_filenames_states_dict.get(filename, UNDEFINED) | ONLY_INC_EXIST
    for filename,  state in curr_merged_filenames_states_dict.items():
        temp_tar_path = os.path.join(curr_merged_tar_dir, filename)
        if state in [ONLY_LAST_V_EXIST, ONLY_INC_EXIST]:
            src_dir = last_v_dir if state == ONLY_LAST_V_EXIST else inc_dir
            temp_src_path = os.path.join(src_dir, filename)
            shutil.copyfile(temp_src_path, temp_tar_path)
        elif state == LAST_V_INC_BOTH_EXIST:
            temp_last_v_path = os.path.join(last_v_dir, filename)
            temp_inc_path = os.path.join(inc_dir, filename)
            merge_data_incremental_order(temp_last_v_path, temp_inc_path, temp_tar_path)


def df_getRepoId_to_labeled_data_col(labeled_data_without_repoid_path, src_issue_body_format_parse_github_id_path,
                                     labeled_data_with_repoid_path, github_repo_link_colname="github_repo_link",
                                     github_repo_id_colname="github_repo_id"):
    df = pd.read_csv(labeled_data_without_repoid_path, index_col=False, encoding="utf-8", dtype=str)
    assert(github_repo_link_colname in df.columns)
    new_columns = []
    for c in df.columns:
        new_columns.append(c)
        if c == github_repo_link_colname:
            new_columns.append(github_repo_id_colname)

    df[github_repo_id_colname] = [""] * len(df)
    df = df[new_columns]
    with open(src_issue_body_format_parse_github_id_path, encoding="utf-8") as f:
        content_str = f.read()

    for idx in df.index:
        pattern = re.compile(f"- (\d+) # repo:{df.loc[idx][github_repo_link_colname]}")
        temp_substrs = re.findall(pattern, content_str)
        temp_substr = temp_substrs[0] if len(temp_substrs) else ""
        df.loc[idx][github_repo_id_colname] = temp_substr

    df.to_csv(labeled_data_with_repoid_path, index=False, encoding="utf-8")
    return


if __name__ == '__main__':
    # prepare data
    BASE_DIR = pkg_rootdir
    labeled_data_filenames = [
        "dbfeatfusion_records_202303_automerged_manulabeled.csv",
        "dbfeatfusion_records_202304_automerged_manulabeled.csv",
    ]
    # dynamic settings
    idx_last_v = 0
    idx_curr_v = 1
    INITIALIZATION = False

    # initialize the source data
    submodule_result_dbfeatfusion_records_dir = os.path.join(BASE_DIR, "db_feature_data_fusion/data/manulabeled")
    database_repo_label_dataframe_dir = os.path.join(BASE_DIR, "data/database_repo_label_dataframe")
    for filename in labeled_data_filenames:
        shutil.copyfile(src=os.path.join(submodule_result_dbfeatfusion_records_dir, filename),
                        dst=os.path.join(database_repo_label_dataframe_dir, filename))
    # default settings
    REGEN_ISSUE_BODY_RAW_STR_LAST_VERSION = True
    ORDER_BY_GITHUB_REPO_LINK = True
    encoding = "utf-8"
    # ------------------------1. Auto generate [data] issue body for opendigger-------------------
    # incremental generation mode
    # 1. auto regenerate last_version
    last_v_labeled_data_filename = labeled_data_filenames[idx_last_v]
    last_v_labeled_data_path = os.path.join(database_repo_label_dataframe_dir, last_v_labeled_data_filename)
    last_v_issue_body_format_txt_path = os.path.join(BASE_DIR, 'data/result/incremental_generation/last_version/issue_body_format.txt')
    last_v_dir = os.path.join(os.path.dirname(last_v_issue_body_format_txt_path), "parsed")
    if REGEN_ISSUE_BODY_RAW_STR_LAST_VERSION:
        df_last_v_labeled_data = pd.read_csv(last_v_labeled_data_path, index_col=False, encoding=encoding)
        if ORDER_BY_GITHUB_REPO_LINK:
            df_last_v_labeled_data.sort_values(by=['github_repo_link'], axis=0, ascending=True, inplace=True)
        auto_gen_issue_body_for_opendigger(df_last_v_labeled_data, last_v_issue_body_format_txt_path,
                                           use_column__mode_col_config=(1, "category_label"), del_emptyval_data=False)

    # 2. generate curr_relative_incremental
    curr_inc_labeled_data_filename = labeled_data_filenames[idx_curr_v]
    curr_inc_labeled_data_path = os.path.join(database_repo_label_dataframe_dir, curr_inc_labeled_data_filename)
    curr_inc_path_issue_body_format_txt = os.path.join(BASE_DIR, 'data/result/incremental_generation/curr_relative_incremental/issue_body_format.txt')
    if not INITIALIZATION:
        df_curr_inc_labeled_data = pd.read_csv(curr_inc_labeled_data_path, index_col=False, encoding=encoding)
        if ORDER_BY_GITHUB_REPO_LINK:
            df_curr_inc_labeled_data.sort_values(by=['github_repo_link'], axis=0, ascending=True, inplace=True)
        auto_gen_issue_body_for_opendigger(df_curr_inc_labeled_data, curr_inc_path_issue_body_format_txt,
                                           use_column__mode_col_config=(1, "category_label"),  # 1 for "category_label_col"
                                           incremental=True, last_v_labeled_data_path=last_v_labeled_data_path)

    # ------------------------2. Create data issue in open-digger--------------------------
    #  e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)
    #  Save content generated with `/parse-github-id` option as "issue_body_format_parse_github_id.txt"
    #  Then turn on parse_github_id_str_to_yaml
    parse_github_id_str_to_yaml = True
    if not parse_github_id_str_to_yaml:
        raise Warning("Please Create data issue in open-digger, then save the bot comments into "
                      "issue_body_format_parse_github_id.txt! Finally, set parse_github_id_str_to_yaml = True.")

    # ----------3. Auto-generate yaml for issue_body_format after parse-github-id----------
    # issue_body_format_parse_github_id.txt is parsed by open-digger, here are steps should be done before:
    #   1) Open a issue(e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)) with content in last_v_issue_body_format_txt_path = './data/issue_body_format.txt'
    #   2) Create an issue comment "/parse-github-id". Bot(github-actions) will reply a parsed format, which will take a while.
    #   3) Copy the parse-github-id content replyed by bot into file src_path = os.path.join(src_dir, "issue_body_format_parse_github_id.txt")
    #   4) Set parse_github_id_prepared = True and run main.py
    #   5) Copy all the generated yaml file into "open-digger/labeled_data/technology/database", replace old files
    #   6) Open a new pull request to [open-digger](https://github.com/X-lab2017/open-digger) to fix the issue created above.
    src_getRepoId_to_yaml_path = os.path.join(last_v_dir, "issue_body_format_parse_github_id.txt")
    tar_getRepoId_to_yaml_dir = last_v_dir
    df_getRepoId_to_yaml(src_getRepoId_to_yaml_path, tar_dir=tar_getRepoId_to_yaml_dir)

    if not INITIALIZATION:
        curr_inc_src_dir = os.path.join(os.path.dirname(curr_inc_path_issue_body_format_txt), "parsed")
        curr_inc_src_path = os.path.join(curr_inc_src_dir, "issue_body_format_parse_github_id.txt")  # manually saved csv: contents are from the open-digger Bot(github-actions) comments
        src_getRepoId_to_yaml_path = curr_inc_src_path
        tar_getRepoId_to_yaml_dir = curr_inc_src_dir
        df_getRepoId_to_yaml(src_getRepoId_to_yaml_path, tar_dir=tar_getRepoId_to_yaml_dir)

        # -------------4. auto generate current_version_incremental_order_merged--------------
        last_version_tar_dir = os.path.join(os.path.dirname(last_v_issue_body_format_txt_path), "parsed")
        curr_merged_tar_dir = os.path.join(BASE_DIR, 'data/result/incremental_generation/current_version_incremental_order_merged')
        auto_gen_current_version_incremental_order_merged(last_version_tar_dir, curr_inc_src_dir, curr_merged_tar_dir, suffix='.yml')
    else:
        # -------------5. get repo id from issue_body_format_parse_github_id.txt as a new column of database repo label dataframe--------------
        labeled_data_without_repoid_path = os.path.join(database_repo_label_dataframe_dir, labeled_data_filenames[idx_last_v])
        labeled_data_with_repoid_filename = labeled_data_without_repoid_path.replace('\\', '/').split('/')[-1].strip('.csv') + '_with_repoid' + '.csv'
        labeled_data_with_repoid_path = os.path.join(database_repo_label_dataframe_dir, labeled_data_with_repoid_filename)
        github_repo_link_colname = "github_repo_link"
        df_getRepoId_to_labeled_data_col(labeled_data_without_repoid_path, src_getRepoId_to_yaml_path,
                                         labeled_data_with_repoid_path, github_repo_link_colname)
