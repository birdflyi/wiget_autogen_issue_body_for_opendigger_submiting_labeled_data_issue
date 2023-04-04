# wiget_autogen_issue_body_for_opendigger_submiting_labeled_data_issue
A wiget  to auto-generate issue_body for `submiting_labeled_data_issue` in [open-digger](https://github.com/X-lab2017/open-digger) repository 

### 1. Auto generate [data] issue body for opendigger
Use INITIALIZATION to control the processing flow of the first version dataset.
Incremental generation mode: The mode refers to 3 directories: last_version, curr_relative_incremental, current_version_incremental_order_merged.
- last_version: last version Opensouce DBMS records, df_last_v.
- curr_relative_incremental: Current version incremental Opensouce DBMS records, deduplicated from concatenate of [df_curr_v, df_last_v, df_last_v].
- current_version_incremental_order_merged: merged data with df_last_v and current_version_incremental records.

Generate issue body format: Function auto_gen_issue_body_for_opendigger will save df_data to a "issue_body_format.txt" in directory last_version or curr_relative_incremental. 
Pattern is controled by dict level_pattern_dict in main.py. See [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245).

### 2. Create data issue in open-digger
e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)
Save content generated with `/parse-github-id` option as "issue_body_format_parse_github_id.txt", then turn on parse_github_id_str_to_yaml.
```python
parse_github_id_str_to_yaml = True
if not parse_github_id_str_to_yaml:
    raise Warning("Please Create data issue in open-digger, then save the bot comments into "
                  "issue_body_format_parse_github_id.txt! Finally, set parse_github_id_str_to_yaml = True.")
```

### 3. Auto-generate yaml for issue_body_format after parse-github-id
issue_body_format_parse_github_id.txt is parsed by open-digger, here are steps should be done before:
  1) Open a issue(e.g. [X-lab2017/open-digger#1245](https://github.com/X-lab2017/open-digger/issues/1245)) with content in last_v_issue_body_format_txt_path = 'xxxdir/issue_body_format.txt'
  2) Create an issue comment "/parse-github-id". Bot(github-actions) will reply a parsed format, which will take a while.
  3) Copy the parse-github-id content replyed by bot into file src_path = os.path.join(src_dir, "issue_body_format_parse_github_id.txt")
  4) Set parse_github_id_prepared = True and run main.py
  5) Copy all the generated yaml file into "open-digger/labeled_data/technology/database", replace old files
  6) Open a new pull request to [open-digger](https://github.com/X-lab2017/open-digger) to fix the issue created above.

Use function df_getRepoId_to_yaml to generate yaml based on "issue_body_format_parse_github_id.txt" with the parse-github-id content replyed by bot.

### 4.auto generate current_version_incremental_order_merged
Use function auto_gen_current_version_incremental_order_merged to merge all the yaml files in directories last_version and curr_relative_incremental into current_version_incremental_order_merged.
