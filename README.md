
# GreyLiterature

Grey literature content quality/user reputation measurement with BERT.

## Running Instructions:

> ``--data_dir`` and ``--num_labels`` are always required.

> If ``--prepare=True`` then ``--raw_path`` is also required.

### To run for a prepared data set:

Set ``--data_dir`` and ``num_labels``

**Example:**

```bash
python main.py --data_dir='./data/design_patterns' --num_labels=12
```

> ``data_dir/`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.


### To prepare and run for a raw CSV file:

Set ``--prepare=True`` and ``--raw_path`` in addition to ``--data_dir`` and ``num_labels``. The prepared data sets will be stored in ``--data_dir``.

**Example:**

```bash
python main.py --data_dir='./data/design_patterns' --num_labels=12 --prepare=True --raw_path='./raw/design_patterns.csv'
```

> Raw CSV files are not divided into training, development, and test sets.

### To run in experiment mode:

Set ``--experiment=True`` in addition to ``--data_dir`` and ``num_labels``.

**Example:**
 
```bash
python main.py --data_dir='./data/design_patterns' --num_labels=12 --experiment=True
```

> It runs for ``train + dev + test = 1000`` samples in the experiment mode.

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)