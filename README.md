# GreyLiterature

Grey literature content quality/user reputation measurement with BERT.

**To run in experiment mode:** 

```python main.py --experiment=True```

> It runs for ``train + dev + test = 1000`` samples in the experiment mode.

**To run for a different CSV file (raw):**

```python main.py --prepare=True --raw_path="<path_to_csv_file>" --data_dir="<path_to_store_prepared_files>"```

> Raw CSV files are not divided into training, development, and test sets.

**To run for a different data set (prepared):**

```python main.py --data_dir="<path_to_data_directory>"```

> ``data_dir/`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)