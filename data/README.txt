COLUMNS:

user_id,question_id,question_title,question_text,answer_id,answer_text,answer_score,
user_answer_count,user_sum_score,user_mean_score,user_median_score,sum_class,mean_class,median_class


DOWNLOAD:
https://drive.google.com/file/d/1OAolz6WbabK7Z439aMM3p0tNV9Q1_xt8



------------------- DP MULTICLASS -------------------

********** BY median_class **********
   median_class  answers  users
0            -1    36246  27239
1             0    12692   1093
2             1     8965    726
3             2     1237    123

********** BY mean_class **********
   mean_class  answers  users
0          -1    36246  27239
1           0     6255    612
2           1     9502    804
3           2     5102    359
4           3     2035    167

********** BY sum_class **********
   sum_class  answers  users
0         -1    36246  27239
1          0      639    108
2          1     2257    349
3          2     5452    672
4          3     5432    457
5          4     4137    226
6          5     2882     92
7          6     2095     38

------------------- DP BINARY -----------------------

***** by user_median_score *****
num of credible answers: 5723 (25% of 22894)
num of credible users: 478 (25% of 1942)
min score of credible users: 2.0
num of not credible answers: 5723 (25% of 22894)
num of not credible users: 518 (27% of 1942)
max score of not credible users: 1.0
num of total answers: 11446
num of total users: 996


***** by user_mean_score *****
num of credible answers: 5723 (25% of 22894)
num of credible users: 430 (22% of 1942)
min score of credible users: 4.19
num of not credible answers: 5723 (25% of 22894)
num of not credible users: 568 (29% of 1942)
max score of not credible users: 1.35
num of total answers: 11446
num of total users: 998


***** by user_sum_score *****
num of credible answers: 5723 (25% of 22894)
num of credible users: 157 (8% of 1942)
min score of credible users: 118
num of not credible answers: 5723 (25% of 22894)
num of not credible users: 850 (44% of 1942)
max score of not credible users: 15
num of total answers: 11446
num of total users: 1007




------------------- SE MULTICLASS -------------------

********** BY median_class **********
   median_class  answers  users
0            -1    28843  19572
1             0    26901   1236
2             1    63788   1793
3             2    31761    714
4             3     1452     98

********** BY mean_class **********
   mean_class  answers  users
0          -1    28843  19572
1           0     5222    512
2           1    28672   1226
3           2    60959   1429
4           3    27770    603
5           4     1279     71

********** BY sum_class **********
   sum_class  answers  users
0         -1    28843  19572
1          0     2385    366
2          1     6024    808
3          2     9693    904
4          3    13668    785
5          4    16903    514
6          5    20851    285
7          6    21646    123
8          7    18559     40
9          8    14173     16

------------------- SE BINARY -----------------------

num of credible answers: 30975 (25% of 123902)
num of credible users: 753 (20% of 3841)
min score of credible users: 4.0
num of not credible answers: 30975 (25% of 123902)
num of not credible users: 1435 (37% of 3841)
max score of not credible users: 2.0
num of total answers: 61950
num of total users: 2188


***** by user_mean_score *****
num of credible answers: 30975 (25% of 123902)
num of credible users: 702 (18% of 3841)
min score of credible users: 8.67
num of not credible answers: 30975 (25% of 123902)
num of not credible users: 1666 (43% of 3841)
max score of not credible users: 3.41
num of total answers: 61950
num of total users: 2368


***** by user_sum_score *****
num of credible answers: 30975 (25% of 123902)
num of credible users: 52 (1% of 3841)
min score of credible users: 2313
num of not credible answers: 30975 (25% of 123902)
num of not credible users: 2833 (74% of 3841)
max score of not credible users: 134
num of total answers: 61950
num of total users: 2885
