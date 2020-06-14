# GreyLiterature

Grey literature answer quality / user reputation measurement with BERT and DistilBERT.

## Running Instructions:

* ``--data_dir`` and ``--labels`` are **always** required.
* Default sequence is ``'TQA'``. Use ``--sequence`` to change.
* Default model is ``'bert'``. Use ``--model`` to change.
* Default device is ``'cpu'``. Use ``--device`` to change.

**To see all arguments and options:**

```bash
python3 main.py --help
```

### Example Run Command:

```bash
python3 main.py --data_dir='data/dp' --labels='median_class' --device='cuda' --model='bert'
```

> :warning: Here, ``--data-dir`` must include ``raw.csv`` that will be divided into train, dev and test sets, and stored under ``data/dp/TQA`` (since the default sequence is ``'TQA'``).

<!--
### 2) To run for a prepared data set:

If you already divided the dataset into train, dev, and test sets, then set only the ``--data_dir`` and ``--labels``.

**Example:**

```bash
python3 main.py --data_dir='./data/dp' --labels='mean_class'
```

> :warning: Here, ``data/dp/TQA`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.


### 2) To run in experiment mode:

Set ``--experiment=True`` in addition to ``--data_dir`` and ``--labels``.

**Example:**
 
```bash
python3 main.py --data_dir='./data/dp' --labels='sum_class' --experiment=True
```

> :warning: Here, the executions will be performed for ``train + dev + test = 1000`` samples.

-->

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)