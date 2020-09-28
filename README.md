# GreyLiterature

Grey literature answer quality / user reputation measurement with BERT and DistilBERT.

## Running Instructions:

* ``--data_dir`` and ``--labels`` arguments are **always** required.
* By default,
  * ``--sequence = 'TQA'`` :arrow_right: Uses Title+Question+Answer sequence. Alternatively, use ``'A'``, ``'TA'``, or ``'QA'``.
  * ``--model = 'bert'`` :arrow_right: Uses BERT model. Alternatively, use ``distilbert``.
  * ``--device = 'cpu'`` :arrow_right: Uses CPU as running device. Alternatively, use ``cuda``.
  * ``--crop = 1.0`` :arrow_right: Uses 100% of the answers and runs multi-class classification. Alternatively, set a value less than ``1.0``. For example, ``--crop = 0.25`` performs binary classification on the top and bottom 25% of the answers.
* To see all arguments and options, run ``python3 main.py --help``

## Example Run Command:

```bash
python3 main.py --model='distilbert' --data_dir='data/dp' --labels='median_class' --device='cuda' --crop=0.25
```

> :warning: Here, ``data/dp`` directory must include ``raw.csv`` file, which will be divided into train, dev and test sets.

> :warning: Since ``--sequence='TQA'`` by default, train, dev and test sets are stored under ``data/dp/TQA``.

> :warning: Since ``crop`` is less than ``1.0``, this command runs a binary classification.

## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)