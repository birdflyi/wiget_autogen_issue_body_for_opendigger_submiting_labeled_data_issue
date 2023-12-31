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

import copy
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
                assert (tn.val["node_body"]["attrrole"] == "elements")
                for elem in temp_data_list:
                    curr_layer_pat_dict[pat_var] = elem
                    content_pattern_formated += format_on_keys(content_pattern_curr_layer, curr_layer_pat_dict, strict=False)
            tn.val["child_node_info"]["childdtype"] = "final"  # reset final after get formated str
        return content_pattern_formated
    else:
        dataset_handler_currlayer_dicts = dataset_handler_layers_dicts[curr_layer]
        assert (len(idx_sortedby_formatable_asc) == len(dataset_handler_currlayer_dicts))
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
                        dataset_handler_unformatable_values[j] = gen_curr_layer_format_str(curr_layer + 1, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict,
                                                                                           temp_dataset_handler_layers_dicts, offset)
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
    assert (FOREST_ROOT_LEVEL_START + 1 == level_start + OFFSET)  # 新增的forest_virtual_root占用了level 0
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
    content_pattern_formated = gen_curr_layer_format_str(curr_layer, bfs_trav_groupby_layer_asc, level_pattern_kasc_dict,
                                                         {curr_layer: data_dict, "del_emptyval_data": del_emptyval_data}, offset=OFFSET)
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
                                f"the step 2. Create data issue in open-digger: save content generated with "
                                f"`/parse-github-id` option as 'issue_body_format_parse_github_id.txt'")
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
            assert (all(df.index == temp_df.index))
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
    issue_body_str = re.sub(r"\n\n+", "\n\n", issue_body_str)
    # print(issue_boddy_str)

    with open(issue_body_format_txt_path, 'w') as f:
        f.write(issue_body_str)


def get_label_records_dict_from_issue_body_format(issue_body_format_path, pre_format_conf, c_sep, c_name_reg,
                                                  rec_k_v_reg, encoding='utf-8'):
    with open(issue_body_format_path, 'r', encoding=encoding) as f:
        s = f.read()
        if pre_format_conf:
            for reg, repl in dict(pre_format_conf).items():
                s = re.sub(r'{}'.format(reg), '{}'.format(repl), s)
        c_rec_formatstrs = s.split(sep=c_sep)
        c_rec_formatstrs = [x.strip() for x in c_rec_formatstrs if x.strip() != '']
        label_records_dict = {}
        for i in range(len(c_rec_formatstrs)):
            c_rec_formatstr = c_rec_formatstrs[i]
            temp_category_label = re.findall(r'{}'.format(c_name_reg), c_rec_formatstr)[0]
            temp_records_dict = dict(re.findall(r'{}'.format(rec_k_v_reg), c_rec_formatstr))
            label_records_dict[temp_category_label] = temp_records_dict
    return label_records_dict


def update_issue_body_format_parse_github_id(ref_body_format_path, update_res_path, last_v_path, curr_inc_path):
    update_res_path = update_res_path or last_v_path
    sep = "#====#"
    all_label_records_parsed_dict, _ = get_all_label_records_parsed_dict_from_parsed_txt([curr_inc_path, last_v_path], sep)

    pre_format_conf_mini = {
        "\n- ([^\\s]+)": "\n\\1",
        "(\n?)Label: ([^\n]+)\n": f"\\1{sep}Label: \\2\n",
    }
    with open(ref_body_format_path, 'r', encoding=encoding) as f:
        s = f.read()
        for reg, repl in dict(pre_format_conf_mini).items():
            s = re.sub(r'{}'.format(reg), '{}'.format(repl), s)
        c_rec_formatstrs = s.split(sep=sep)
        for i in range(len(c_rec_formatstrs)):
            c_rec_formatstr = c_rec_formatstrs[i]
            c_record_keys = c_rec_formatstr.split('\n')
            for j in range(len(c_record_keys)):
                key = c_record_keys[j]
                if key in all_label_records_parsed_dict.keys():
                    c_record_keys[j] = all_label_records_parsed_dict[key]
            c_rec_formatstr_parse_github_id = '\n'.join(c_record_keys)
            c_rec_formatstrs[i] = c_rec_formatstr_parse_github_id
        c_rec_formatstrs_parse_github_id = ''.join(c_rec_formatstrs)
    with open(update_res_path, 'w', encoding=encoding) as f:
        f.write(c_rec_formatstrs_parse_github_id)
    print(f"{update_res_path} is saved!")
    return


def get_all_label_records_parsed_dict_from_parsed_txt(parsed_txt_paths, sep="#====#"):
    pre_format_conf_kpattern_vstdreplacement = {
        "\n(- \\d+ # repo:([^\\s]+))": "\nREC::\\2[k:v]\\1",
        "\n(- ([^\\s]+) # not found)": "\nREC::\\2[k:v]\\1",
        "\n?Label: ([^\n]+)\n": f"\n{sep}name: Database - \\1\n",
    }
    pre_format_conf = pre_format_conf_kpattern_vstdreplacement
    c_name_reg = "name: Database - ([^\n]+)\n"
    rec_k_v_reg = "REC::([^\\s]+)\[k:v\](- [^\\n]+)"
    if type(parsed_txt_paths) is str:
        parsed_txt_paths = [parsed_txt_paths]
    else:
        parsed_txt_paths = list(parsed_txt_paths)

    top_level_keys = []
    d1 = {}
    for parsed_txt_path in parsed_txt_paths:
        temp_label_records_dict = get_label_records_dict_from_issue_body_format(parsed_txt_path, pre_format_conf, c_sep=sep,
                                                                                c_name_reg=c_name_reg,
                                                                                rec_k_v_reg=rec_k_v_reg)
        top_level_keys += [k for k in temp_label_records_dict.keys() if k not in top_level_keys]
        d2 = temp_label_records_dict
        for k in top_level_keys:
            if k in d1.keys():
                d1[k].update(d2.get(k, {}))
            else:
                d1[k] = d2.get(k, {})

    all_label_records_2_layer_parsed_dict = d1
    all_label_records_parsed_dict = {}
    for d in all_label_records_2_layer_parsed_dict.values():
        all_label_records_parsed_dict.update(d)
    return all_label_records_parsed_dict, all_label_records_2_layer_parsed_dict


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
        "name: Database - Vector DBMS": "name: Database - Vector",

    }
    filename_reg = "name: Database - ([^\n]+)\n"
    to_yaml_parts(src_path=src_path, tar_dir=tar_dir, pre_format_conf=pre_format_conf_kpattern_vstdreplacement,
                  filename_reg=filename_reg, sep=sep)
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


def merge_data_incremental_order(last_v_path, inc_path, tar_path, keep_list=None, encoding='utf-8', **kwargs):
    with open(last_v_path, 'r', encoding=encoding) as f:
        last_v_yaml_formatstr = f.read()
    with open(inc_path, 'r', encoding=encoding) as f:
        inc_yaml_formatstr = f.read()
    github_repo_text_header = "\n  github_repo:"
    github_repo_text_paragraph_pattern = r"(\n  github_repo:((\n    - [^\n]*)*))"
    not_found_str = " # not found"
    drop_not_found = kwargs.get("drop_not_found", True)
    last_v_yaml_data_github_repo_formatstr_matchlist = re.findall(github_repo_text_paragraph_pattern, last_v_yaml_formatstr)[0]
    inc_yaml_data_github_repo_formatstr_matchlist = re.findall(github_repo_text_paragraph_pattern, inc_yaml_formatstr)[0]
    repos_delimiter = '\n    - '
    with open(tar_path, 'w', encoding=encoding) as f:
        # head
        tar_text = re.sub(github_repo_text_paragraph_pattern, github_repo_text_header + '{records}', last_v_yaml_formatstr)
        # tar_text += github_repo_text_header
        # merge records
        last_v_repos_parsed_records = last_v_yaml_data_github_repo_formatstr_matchlist[1].split(repos_delimiter)
        last_v_repos_parsed_records = list(filter(None, last_v_repos_parsed_records))
        last_v_repos_parsed_keys = [re.sub(r"\d+ #[ \w]+:", '', s) for s in last_v_repos_parsed_records]
        last_v_repos_parsed_dict = dict(zip(last_v_repos_parsed_keys, last_v_repos_parsed_records))

        inc_repos_parsed_records = inc_yaml_data_github_repo_formatstr_matchlist[1].split(repos_delimiter)
        inc_repos_parsed_records = list(filter(None, inc_repos_parsed_records))
        inc_repos_parsed_keys = [re.sub(r"\d+ #[ \w]+:", '', s) for s in inc_repos_parsed_records]
        inc_repos_parsed_keys = [s.replace(not_found_str, "") for s in inc_repos_parsed_keys]
        inc_repos_parsed_dict = dict(zip(inc_repos_parsed_keys, inc_repos_parsed_records))

        try:
            merged_repos_parsed_dict = dict(**last_v_repos_parsed_dict, **inc_repos_parsed_dict)
        except TypeError:
            merged_repos_parsed_dict = copy.deepcopy(last_v_repos_parsed_dict)
            merged_repos_parsed_dict.update(inc_repos_parsed_dict)

        if drop_not_found:
            merged_repos_parsed_dict = {k: v for k, v in merged_repos_parsed_dict.items() if not str(v).__contains__(not_found_str)}
        merged_repos_parsed_dict_sorted = dict(sorted(merged_repos_parsed_dict.items(), key=lambda x: x[0], reverse=False))

        if keep_list is not None:
            yaml_data_github_repos_checked = [v for k, v in merged_repos_parsed_dict_sorted.items() if k in keep_list]
        else:
            yaml_data_github_repos_checked = list(merged_repos_parsed_dict_sorted.values())
        yaml_data_github_repo_formatstr = repos_delimiter + repos_delimiter.join(yaml_data_github_repos_checked)
        tar_text = tar_text.format(**{"records": yaml_data_github_repo_formatstr})
        f.seek(0, 0)
        f.write(tar_text)
        print(f'Warning: {tar_path} is updated! Manual check may be required!!!')


def order_github_repo_by_github_repo_link(yaml_path, ascending=True):
    sep = "#====#"
    yaml_format_conf_mini = {
        "\n  ([^\n]+):\n": f"\n{sep}  \\1:\n"
    }
    c_name_reg = "  ([^\n]+):\n"
    record_key_reg = "    - \d+ # {c_name}:([^\\s]+)"
    category_name_comment_pattern_dict = {
        "github_repo": "repo",
        "github_user": "user",
    }
    data_dict = {}
    with open(yaml_path, 'r', encoding=encoding) as f:
        s = f.read()
        for reg, repl in dict(yaml_format_conf_mini).items():
            s = re.sub(r'{}'.format(reg), '{}'.format(repl), s)
        c_rec_formatstrs = s.split(sep=sep)
        for i in range(len(c_rec_formatstrs)):
            c_rec_formatstr = c_rec_formatstrs[i]
            temp_category_label_findall = re.findall(r'{}'.format(c_name_reg), c_rec_formatstr)
            if not temp_category_label_findall:
                continue

            temp_category_label = str(temp_category_label_findall[0])
            if temp_category_label in category_name_comment_pattern_dict.keys():
                c_name = category_name_comment_pattern_dict[temp_category_label]
            elif temp_category_label.startswith("github_"):
                c_name = str(temp_category_label).removesuffix("github_")
            else:
                raise ValueError(f"Cant identify the comment pattern of category {temp_category_label} in {yaml_path}!")
            record_key_reg = record_key_reg.format(c_name=c_name)
            data_dict[temp_category_label] = {}
            temp_c_items_key_newrec_dict = data_dict[temp_category_label]
            c_record_items = c_rec_formatstr.split('\n')
            c_rec_formatstr_parse_github_id = ""
            for j in range(len(c_record_items)):
                c_record_keys = re.findall(r'{}'.format(record_key_reg), c_record_items[j])
                if not c_record_keys:
                    if not c_rec_formatstr_parse_github_id:
                        c_rec_formatstr_parse_github_id = c_record_items[j]
                    else:
                        if len(temp_c_items_key_newrec_dict):
                            newrec_dict_sorted_by_key = dict(sorted(temp_c_items_key_newrec_dict.items(),
                                                                    key=lambda x: x[0], reverse=not ascending))
                            c_rec_formatstr_parse_github_id += '\n' + '\n'.join(newrec_dict_sorted_by_key.values())
                            temp_c_items_key_newrec_dict = {}  # clear up after saved into c_rec_formatstr_parse_github_id
                        c_rec_formatstr_parse_github_id += '\n' + c_record_items[j]
                else:
                    c_record_key = c_record_keys[0]
                    temp_c_items_key_newrec_dict[c_record_key] = c_record_items[j]

            if len(temp_c_items_key_newrec_dict):
                newrec_dict_sorted_by_key = dict(sorted(temp_c_items_key_newrec_dict.items(),
                                                        key=lambda x: x[0], reverse=not ascending))
                c_rec_formatstr_parse_github_id += '\n' + '\n'.join(newrec_dict_sorted_by_key.values())
                temp_c_items_key_newrec_dict = {}
            c_rec_formatstrs[i] = c_rec_formatstr_parse_github_id
        c_rec_formatstrs_parse_github_id = ''.join(c_rec_formatstrs)
    with open(yaml_path, 'w', encoding=encoding) as f:
        f.write(c_rec_formatstrs_parse_github_id)
    print(f"{yaml_path} is saved!")
    return


def auto_gen_current_version_incremental_order_merged(last_v_dir, inc_dir, curr_merged_tar_dir, suffix='.yml',
                                                      redundancy_check_df=None, **kwargs):
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
            if redundancy_check_df is None:
                keep_list = None
            else:
                curr_category = str(filename).removesuffix(suffix)
                temp_df = pd.DataFrame(redundancy_check_df)[["github_repo_link", "category_label"]]
                belong_to_curr_category = lambda x: str(x).replace("#dbdbio>|<dbengines#", ",").replace(" DBMS", "").\
                    replace(" ", "_").replace("-", "_").lower().__contains__(curr_category)
                temp_df_checked = temp_df[temp_df["category_label"].apply(belong_to_curr_category).values]
                keep_list = temp_df_checked["github_repo_link"].values
            merge_data_incremental_order(temp_last_v_path, temp_inc_path, temp_tar_path, keep_list=keep_list, **kwargs)
        else:  # should never happen
            continue
        # order_github_repo_by_github_repo_link(temp_tar_path, ascending=True)


def df_getRepoId_to_labeled_data_col(labeled_data_without_repoid_path, all_label_records_parsed_dict,
                                     labeled_data_with_repoid_path, github_repo_link_colname="github_repo_link",
                                     github_repo_id_colname="github_repo_id"):
    df = pd.read_csv(labeled_data_without_repoid_path, index_col=False, encoding="utf-8", dtype=str)
    assert (github_repo_link_colname in df.columns)
    new_columns = []
    for c in df.columns:
        new_columns.append(c)
        if c == github_repo_link_colname:
            new_columns.append(github_repo_id_colname)

    df[github_repo_id_colname] = [""] * len(df)
    df = df[new_columns]

    for idx in df.index:
        temp_github_repo_link = df.loc[idx][github_repo_link_colname]
        if temp_github_repo_link not in all_label_records_parsed_dict.keys():
            df.loc[idx][github_repo_id_colname] = ""
        else:
            temp_content_str = all_label_records_parsed_dict[temp_github_repo_link]
            pattern = re.compile(f"- (\d+) # repo:{df.loc[idx][github_repo_link_colname]}")
            temp_substrs = re.findall(pattern, temp_content_str)
            temp_substr = temp_substrs[0] if len(temp_substrs) else ""
            df.loc[idx][github_repo_id_colname] = temp_substr

    df.to_csv(labeled_data_with_repoid_path, index=False, encoding="utf-8")
    return


if __name__ == '__main__':
    # Step 0. Update git submodules
    # git command in bash:
    # $ git submodule foreach git checkout main
    # $ git submodule foreach git pull

    # prepare data
    BASE_DIR = pkg_rootdir
    # Step 1. Add new data filenames as input dataset.
    labeled_data_filenames = [
        "dbfeatfusion_records_202303_automerged_manulabeled.csv",
        "dbfeatfusion_records_202304_automerged_manulabeled.csv",
        "dbfeatfusion_records_202305_automerged_manulabeled.csv",
        "dbfeatfusion_records_202306_automerged_manulabeled.csv",
        "dbfeatfusion_records_202307_automerged_manulabeled.csv",
        "dbfeatfusion_records_202308_automerged_manulabeled.csv",
        "dbfeatfusion_records_202309_automerged_manulabeled.csv",
        "dbfeatfusion_records_202310_automerged_manulabeled.csv",
        "dbfeatfusion_records_202311_automerged_manulabeled.csv",
        "dbfeatfusion_records_202312_automerged_manulabeled.csv",
    ]
    # dynamic settings
    idx_last_v = -2
    idx_curr_v = -1

    # Step 2. Change the curr_stage from 0 to 2 before run main.py and solve the warnings related to data.
    curr_stage = 2

    # static settings
    STAGE__UPDATE_LAST_VERSION__SAVE_PARSED = {
        0: [True, False, False],  # update the last version parsed data
        1: [False, False, False],  # auto generate the issue body(see Warning in 2. Create data issue in open-digger) for df_curr_inc_labeled_data
        # Resolve the warning by creating data issue in open-digger and save parsed data before next step!
        2: [False, True, True]  # split the curr_inc_issue_body_format_parse_github_id into yml files
    }
    UPDATE_LAST_VERSION = STAGE__UPDATE_LAST_VERSION__SAVE_PARSED[curr_stage][0]
    save_parsed_as_curr_inc_issue_body_format_parse_github_id = STAGE__UPDATE_LAST_VERSION__SAVE_PARSED[curr_stage][1]

    # initialize the source data
    INITIALIZE_DATABASE_REPO_LABEL_DATAFRAME = True
    submodule_result_dbfeatfusion_records_dir = os.path.join(BASE_DIR, "db_feature_data_fusion/data/manulabeled")
    database_repo_label_dataframe_dir = os.path.join(BASE_DIR, "data/database_repo_label_dataframe")
    if INITIALIZE_DATABASE_REPO_LABEL_DATAFRAME:
        for filename in labeled_data_filenames:
            shutil.copyfile(src=os.path.join(submodule_result_dbfeatfusion_records_dir, filename),
                            dst=os.path.join(database_repo_label_dataframe_dir, filename))
    # default settings
    ORDER_BY_GITHUB_REPO_LINK = True
    encoding = "utf-8"
    # ------------------------1. Auto generate [data] issue body for opendigger-------------------
    # incremental generation mode
    # 1. auto regenerate last_version
    REGEN_ISSUE_BODY_RAW_STR_LAST_VERSION = UPDATE_LAST_VERSION
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
    df_curr_inc_labeled_data = pd.read_csv(curr_inc_labeled_data_path, index_col=False, encoding=encoding)
    if UPDATE_LAST_VERSION:  # Block this process when last version still needs to be updated.
        pass
    else:  # Merge current incremental dataframe and last version labeled data into curr_inc_path_issue_body_format_txt
        if ORDER_BY_GITHUB_REPO_LINK:
            df_curr_inc_labeled_data.sort_values(by=['github_repo_link'], axis=0, ascending=True, inplace=True)
        auto_gen_issue_body_for_opendigger(df_curr_inc_labeled_data, curr_inc_path_issue_body_format_txt,
                                           use_column__mode_col_config=(1, "category_label"),  # 1 for "category_label_col"
                                           incremental=True, last_v_labeled_data_path=last_v_labeled_data_path)

    # ------------------------2. Create data issue in open-digger--------------------------
    #  e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)
    #  Save content generated with `/parse-github-id` option as "issue_body_format_parse_github_id.txt"
    #  Then set save_parsed_as_curr_inc_issue_body_format_parse_github_id = True
    last_v_issue_body_format_parse_github_id_path = os.path.join(last_v_dir, "issue_body_format_parse_github_id.txt")
    curr_inc_issue_body_format_txt_path = os.path.join(BASE_DIR, 'data/result/incremental_generation/curr_relative_incremental/issue_body_format.txt')
    curr_inc_dir = os.path.join(os.path.dirname(curr_inc_issue_body_format_txt_path), "parsed")
    curr_inc_issue_body_format_parse_github_id_path = os.path.join(curr_inc_dir, "issue_body_format_parse_github_id.txt")
    if UPDATE_LAST_VERSION:
        update_issue_body_format_parse_github_id(last_v_issue_body_format_txt_path,
                                                 last_v_issue_body_format_parse_github_id_path,
                                                 last_v_path=last_v_issue_body_format_parse_github_id_path,
                                                 curr_inc_path=curr_inc_issue_body_format_parse_github_id_path)
    elif not save_parsed_as_curr_inc_issue_body_format_parse_github_id:
        raise Warning(f"Please Create data issue in open-digger with contents in {curr_inc_issue_body_format_txt_path}, "
                      f"then save the bot comments into {curr_inc_issue_body_format_parse_github_id_path}! "
                      f"\t\nFinally, set parse_github_id_str_to_yaml = True.")
    else:
        pass

    # ----------3. Auto-generate yaml for issue_body_format after parse-github-id----------
    # issue_body_format_parse_github_id.txt is parsed by open-digger, here are steps should be done before:
    #   1) Open an issue(e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)) with contents in curr_inc_issue_body_format_txt_path = './issue_body_format.txt'
    #   2) Create an issue comment "/parse-github-id". Bot(github-actions) will reply a parsed format, which will take a while.
    #   3) Copy the parse-github-id content replyed by bot into file "issue_body_format_parse_github_id.txt"
    #   4) Set save_parsed_as_curr_inc_issue_body_format_parse_github_id = True and run main.py
    #   5) Copy all the generated yaml file into "open-digger/labeled_data/technology/database", replace old files
    #   6) Open a new pull request to [open-digger](https://github.com/X-lab2017/open-digger) to fix the issue created above.
    src_last_v_parsed_txt_path = os.path.join(last_v_dir, "issue_body_format_parse_github_id.txt")
    tar_last_v_parsed_txt_dir = last_v_dir

    curr_inc_src_dir = os.path.join(os.path.dirname(curr_inc_path_issue_body_format_txt), "parsed")
    src_curr_inc_parsed_txt_path = os.path.join(curr_inc_src_dir, "issue_body_format_parse_github_id.txt")  # manually saved csv: contents are from the open-digger Bot(github-actions) comments
    tar_curr_inc_parsed_txt_dir = curr_inc_src_dir
    if UPDATE_LAST_VERSION:
        df_getRepoId_to_yaml(src_last_v_parsed_txt_path, tar_dir=tar_last_v_parsed_txt_dir)
    else:
        for filename in os.listdir(tar_curr_inc_parsed_txt_dir):  # remove the last version incremental data
            if filename.endswith('.yml'):
                os.remove(os.path.join(tar_curr_inc_parsed_txt_dir, filename))
        df_getRepoId_to_yaml(src_curr_inc_parsed_txt_path, tar_dir=tar_curr_inc_parsed_txt_dir)

        # -------------4. auto generate current_version_incremental_order_merged--------------
        last_version_tar_dir = os.path.join(os.path.dirname(last_v_issue_body_format_txt_path), "parsed")
        curr_merged_tar_dir = os.path.join(BASE_DIR, 'data/result/incremental_generation/current_version_incremental_order_merged')
        for filename in os.listdir(curr_merged_tar_dir):  # remove the last version incremental data
            if filename.endswith('.yml'):
                os.remove(os.path.join(curr_merged_tar_dir, filename))
        DROP_NOT_FOUND = True
        auto_gen_current_version_incremental_order_merged(last_version_tar_dir, curr_inc_src_dir, curr_merged_tar_dir,
                                                          suffix='.yml', redundancy_check_df=df_curr_inc_labeled_data,
                                                          drop_not_found=DROP_NOT_FOUND)

    # -------------5. get repo id from issue_body_format_parse_github_id.txt as a new column of database repo label dataframe--------------
    idx_v = idx_last_v if UPDATE_LAST_VERSION else idx_curr_v
    take_parsed_repo_id_as_df_new_col = STAGE__UPDATE_LAST_VERSION__SAVE_PARSED[curr_stage][2]
    labeled_data_without_repoid_path = os.path.join(database_repo_label_dataframe_dir, labeled_data_filenames[idx_v])
    labeled_data_with_repoid_filename = labeled_data_without_repoid_path.replace('\\', '/').split('/')[-1].strip('.csv') + '_with_repoid' + '.csv'
    labeled_data_with_repoid_path = os.path.join(database_repo_label_dataframe_dir, labeled_data_with_repoid_filename)
    all_label_records_parsed_dict, _ = get_all_label_records_parsed_dict_from_parsed_txt([src_last_v_parsed_txt_path, src_curr_inc_parsed_txt_path])
    if take_parsed_repo_id_as_df_new_col:
        df_getRepoId_to_labeled_data_col(labeled_data_without_repoid_path, all_label_records_parsed_dict,
                                         labeled_data_with_repoid_path, github_repo_link_colname="github_repo_link")
