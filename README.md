# wiget_autogen_issue_body_for_opendigger_submiting_labeled_data_issue
A wiget  to auto-generate issue_body for `submiting_labeled_data_issue` in [open-digger](https://github.com/X-lab2017/open-digger) repository 

Use the git command to update submodules:

```bash
$ git clone git@github.com:birdflyi/wiget_autogen_issue_body_for_opendigger_submiting_labeled_data_issue.git
$ cd wiget_autogen_issue_body_for_opendigger_submiting_labeled_data_issue/db_feature_data_fusion/
$ git pull
$ git submodule deinit --all
$ git submodule sync --recursive
$ git submodule update --init --recursive
```

### 1. Auto generate [data] issue body for opendigger
Use `curr_stage` to control the processing flow of the first version dataset.
Incremental generation mode: The mode refers to 3 directories: last_version, curr_relative_incremental, current_version_incremental_order_merged.

- last_version: last version Opensouce DBMS records, df_last_v.
- curr_relative_incremental: Current version incremental Opensouce DBMS records, deduplicated from concatenate of [df_curr_v, df_last_v, df_last_v].
- current_version_incremental_order_merged: merged data with df_last_v and current_version_incremental records.

Generate issue body format: Function auto_gen_issue_body_for_opendigger will save df_data to a "issue_body_format.txt" in directory last_version or curr_relative_incremental. 
Pattern is controled by dict level_pattern_dict in main.py. See [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245).

### 2. Preparations need to be done before stage 2
#### 2.1 Create data issue in open-digger
e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)
Save content generated with `/parse-github-id` option as "issue_body_format_parse_github_id.txt", then turn on save_parsed_as_curr_inc_issue_body_format_parse_github_id in stage 2.

#### 2.2 Auto-generate yaml for issue_body_format after parse-github-id
issue_body_format_parse_github_id.txt is parsed by open-digger, here are steps should be done before:
  1) Open an issue(e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)) with contents in curr_inc_issue_body_format_txt_path = './issue_body_format.txt'
  2) Create an issue comment "/parse-github-id". Bot(github-actions) will reply a parsed format, which will take a while.
  3) Copy the parse-github-id content replyed by bot into file "issue_body_format_parse_github_id.txt"
  4) Set save_parsed_as_curr_inc_issue_body_format_parse_github_id = True and run main.py
  5) Copy all the generated yaml file into "open-digger/labeled_data/technology/database", replace old files
  6) Open a new pull request to [open-digger](https://github.com/X-lab2017/open-digger) to fix the issue created above.

Use function df_getRepoId_to_yaml to generate yaml based on "issue_body_format_parse_github_id.txt" with the parse-github-id content replyed by bot.

### 3.auto generate current_version_incremental_order_merged
Use function auto_gen_current_version_incremental_order_merged to merge all the yaml files in directories last_version and curr_relative_incremental into current_version_incremental_order_merged.
The function runs in stage 2 by setting take_parsed_repo_id_as_df_new_col = True.
