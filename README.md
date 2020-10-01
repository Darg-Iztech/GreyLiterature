# GreyLiterature

Grey literature answer quality / user reputation measurement with BERT and DistilBERT.

## Training models using DP, SE, or other datasets

### Dataset:
When training a model ``--data_dir`` and ``--labels`` arguments are **always** required.
* ``--data_dir`` :arrow_right: The directory including ``raw.csv`` file, which will be divided into train, dev and test sets. The ``raw.csv`` files for DP and SE are available at [IZTECH Cloud Repository](https://cloud.iyte.edu.tr/index.php/s/WuI9JTTgvW2dwAc).
* ``--labels`` :arrow_right: The argument that must be set as the name of the column containing the ground truth values for the author classes. Typically, it is set as ``'sum_class'``, ``'mean_class'``, or ``'median_class'``.

When working on other data sets, be sure to organize the ``raw.csv`` file to include the following columns:

```
user_id,question_title,question_text,answer_text,answer_score,user_answer_count,sum_class,mean_class,median_class
```

To see a sample ``raw.csv`` file :arrow_right: [data/test/raw.csv](https://github.com/Darg-Iztech/GreyLiterature/blob/master/data/test/raw.csv)

### Options and arguments:

Some default training arguments are:
* ``--sequence = 'TQA'`` :arrow_right: Uses Title+Question+Answer sequence. Alternatively, use ``'A'``, ``'TA'``, or ``'QA'``.
* ``--model = 'bert'`` :arrow_right: Uses BERT model. Alternatively, use ``distilbert``.
* ``--device = 'cpu'`` :arrow_right: Uses CPU as running device. Alternatively, use ``cuda``.
* ``--crop = 1.0`` :arrow_right: Uses 100% of the answers and runs multi-class classification. Alternatively, set a value less than ``1.0``. For example, ``--crop = 0.25`` performs binary classification on the top and bottom 25% of the answers.

To see all training arguments and options, run:
```
python3 main.py --help
```

### Example Training Command:

```bash
python3 main.py --model='distilbert' --data_dir='data/dp' --labels='median_class' --device='cuda' --crop=0.25
```

> :warning: Here, ``data/dp`` directory must include ``raw.csv`` file, which will be divided into train, dev and test sets.

> :warning: Since ``--sequence='TQA'`` by default, train, dev and test sets are stored under ``data/dp/TQA``.

> :warning: Since ``crop`` is less than ``1.0``, this command runs a binary classification.

## Testing with a Pre-trained Model

**Step 1)** Download the pre-trained model with 71.6% accuracy from [HERE]() (~1.3 Gb ``pth.tar`` file).

**Step 2)** Use ``predict.py`` module to predict the reputability of specific author(s).

The ``predict.py`` module takes 2 arguments:
* ``--checkpoint_path`` (required) :arrow_right: Path to ``pth.tar`` file downloaded in Step 1.
* ``--test_path`` (optional) :arrow_right: Path to the CSV file including questions and answers of which author reputability will be predicted. The file must include ``question,answer,label`` columns. If the ``--test_path`` is not set, the question, answer, and label are taken as console input.

To see a sample ``test.csv`` file :arrow_right: [data/test/test.csv](https://github.com/Darg-Iztech/GreyLiterature/blob/master/data/test/test.csv)

### Example Testing Commands:

**Example 1)** ``--test_path`` is set to ``test.csv`` file that includes 3 answers from 3  authors:
```bash
python3 predict.py --checkpoint_path='models/checkpoint.pth.tar' --test_path='data/test/test.csv'
------------------
Expected:  [0 1 0]
Predicted: [1 1 0]
```

**Example 2)** ``--test_path`` is not set:
```bash
python3 predict.py --checkpoint_path='models/checkpoint.pth.tar'
------------------
Enter question (as plain text): Which kind of Singleton is this?
Enter answer (as plain text): It is not a singleton. It is multiton pattern.
Enter label (as 0 or 1): 1
------------------
Expected:  [1]
Predicted: [1]
```


## References

Modified version of the code in [https://github.com/isspek/west\_iyte\_plausability\_news\_detection](https://github.com/isspek/west_iyte_plausability_news_detection)