# GreyLiterature

Grey literature answer quality / user reputation measurement with BERT and DistilBERT.

## Downloading the datasets stored in Git LFS:

> :warning: To retrieve ``raw.csv`` files with ``git clone`` or ``git pull`` make sure Git LFS is pre-installed in your environment.

## Running Instructions:

> :warning: ``--data_dir`` is always required.

<!--If ``--prepare=True`` then ``--raw_path`` is also required.-->

> :warning: Runs in ``'TQA'`` mode by default. Use ``--mode`` to change concatenation mode.

> :warning: Runs using ``'bert'`` model by default. Use ``--model`` to change language model.

> :warning: Runs on ``'cpu'`` by default. Use ``--device`` to change device.

### :arrow_forward: To prepare and run for a raw CSV file:

<!--Set ``--prepare=True`` and ``--raw_path`` in addition to ``--data_dir`` and ``num_labels``. Also set ``--mode`` if needed.-->

Set ``--data_dir`` and ``--prepare=True``.

> :warning: ``--data-dir`` must include ``raw.csv`` which is not divided into train, dev and test sets yet.

> :warning: The prepared data sets will be stored according to ``--data_dir`` and ``--mode`` (which is ``'TQA'`` by default).

**Example:**

```bash
python3 main.py --data_dir='data/dp' --prepare=True
```

In this example, ``data/dp/raw.csv`` file will be divided into train, dev and test sets, and stored separately under ``data/dp/TQA`` (since the default mode is ``'TQA'``).

### :arrow_forward: To run for a prepared data set:

If you already divided the dataset into train, dev, and test sets, then set only the ``--data_dir``. In the example below, we also set ``--mode``, which is optional.

**Example:**

```bash
python3 main.py --mode='TA' --data_dir='./data/dp'
```

In this example, ``data_dir/TA`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.

### :arrow_forward: To run using DistilBERT:

Set ``--model='distilbert'`` in addition to ``--data_dir``.

```bash
python3 main.py --model='distilbert' --data_dir='./data/dp'
```

### :arrow_forward: To run in experiment mode:

Set ``--experiment=True`` in addition to ``--data_dir``.

**Example:**
 
```bash
python3 main.py --data_dir='./data/dp' --experiment=True
```

In this example, the executions will be performed for ``train + dev + test = 1000`` samples.

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)