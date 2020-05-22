# GreyLiterature

Grey literature content quality/user reputation measurement with BERT.

## Running Instructions:

> :warning: ``--data_dir`` and ``--num_labels`` are always required.

<!--If ``--prepare=True`` then ``--raw_path`` is also required.-->

> :warning: Runs in ``'TA'`` mode by default. Use ``--mode`` to change concatenation mode.

> :warning: Runs using ``'bert'`` model by default. Use ``--model`` to change language model.

### :arrow_forward: To run for a prepared data set:

Set ``--data_dir`` and ``--num_labels``. Also set ``--mode`` if needed.

**Example:**

```bash
python main.py --mode='TA' --data_dir='./data/design_patterns' --num_labels=12
```

In this example, ``data_dir/TA`` must include ``train.tsv``, ``dev.tsv``, and ``test.tsv``.


### :arrow_forward: To prepare and run for a raw CSV file:

<!--Set ``--prepare=True`` and ``--raw_path`` in addition to ``--data_dir`` and ``num_labels``. Also set ``--mode`` if needed.-->

Set ``--prepare=True`` in addition to ``--data_dir`` and ``num_labels``. Also set ``--mode`` if needed.

> :warning: ``--data-dir`` must include ``raw.csv`` which is not divided into train, dev and test sets yet.

> :warning: The prepared data sets will be stored according to ``--data_dir`` and ``--mode``. It is ``./data/design_patternts/TA`` for the example below.

**Example:**

```bash
python main.py --data_dir='./data/design_patterns' --num_labels=12 --prepare=True
```

In this example, ``./data/design_patterns/raw.csv`` file will be divided into train, dev and test sets, and stored separately under ``./data/design_patterns/TA``.

### :arrow_forward: To run using DistilBERT:

Set ``--model='distilbert'`` in addition to ``--data_dir`` and ``num_labels``. Also set ``--mode`` if needed.

```bash
python main.py --model='distilbert' --data_dir='./data/design_patterns' --num_labels=12
```

### :arrow_forward: To run in experiment mode:

Set ``--experiment=True`` in addition to ``--data_dir`` and ``num_labels``. Also set ``--mode`` if needed.

**Example:**
 
```bash
python main.py --data_dir='./data/design_patterns' --num_labels=12 --experiment=True
```

In this example, the executions will be performed for ``train + dev + test = 1000`` samples.

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)