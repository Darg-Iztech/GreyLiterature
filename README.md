# GreyLiterature

Grey literature answer quality / user reputation measurement with BERT and DistilBERT.

## Downloading the datasets stored in Git LFS:

To retrieve ``raw.csv`` files with ``git clone`` or ``git pull`` make sure Git LFS is pre-installed in your environment.

## Running Instructions:

* ``--data_dir`` is always required.
* Default sequence is ``'TQA'``. Use ``--sequence`` to change.
* Default model is ``'bert'``. Use ``--model`` to change.
* Default device is ``'cpu'``. Use ``--device`` to change.

**To see all arguments:**

```bash
python3 main.py --help
```

### 1) To prepare and run for a raw CSV file:

Set ``--data_dir`` and ``--prepare=True``.

**Example:**

```bash
python3 main.py --data_dir='data/dp' --prepare=True
```

> :warning: Here, ``--data-dir`` must include ``raw.csv`` that will be divided into train, dev and test sets, and stored under ``data/dp/TQA`` (since the default sequence is ``'TQA'``).

### 2) To run for a prepared data set:

If you already divided the dataset into train, dev, and test sets, then set only the ``--data_dir``.

**Example:**

```bash
python3 main.py --data_dir='./data/dp'
```

> :warning: Here, ``data/dp/TQA`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.

### 3) To run in experiment mode:

Set ``--experiment=True`` in addition to ``--data_dir``.

**Example:**
 
```bash
python3 main.py --data_dir='./data/dp' --experiment=True
```

> :warning: Here, the executions will be performed for ``train + dev + test = 1000`` samples.

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)