# Mini Math Solver

A small language model capable of solving math word problems

When trained on a single GPU for one hour with only 900 example word problems, a 250M-parameter *mini-math-solver* acquires step-by-step reasoning ability and achieves 59% accuracy on an unseen 100-problem test set

Empirical evidence shows that large language models perform well when instructed to "show their work" and explain their reasoning. *mini-math-solver* demonstrates that the same principle applies to small models as well

During training, *mini-math-solver* is instructed to produce two outputs: 

1. A numeric answer to a problem based on a ground truth label
2. An explanation of its answer using an example written by a larger language model

A loss function measures how well it performs on the two tasks. By minimizing this loss, *mini-math-solver* improves its performance on both objectives

Once *mini-math-solver*'s weights acquire an ability to explain answers, accurate problem-solving is quickly unlocked

### Samples

The following word problems were given as inputs to *mini-math-solver*. The answers and explanations were generated entirely by the model

______

Julia played soccer with 2 friends on Monday, 14 friends on Tuesday, and 16 friends on Wednesday. How many friends did she play with on Tuesday and Wednesday? 

*mini-math-solver:*

**Answer:** ( 14.0 + 16.0 )

**Explanation:** Julia played with 14 friends on tuesday. She played with 16 friends on wednesday. So she played with 14 + 16
______

Paco had 39 sweet cookies and 6 salty cookies. He ate 23 salty cookies and 32 sweet cookies. How many more sweet cookies than salty cookies did he eat?

*mini-math-solver:*

**Answer:** ( 32.0 - 23.0 )

**Explanation:** Paco ate 23 salty cookies and 32 sweet cookies. So he ate 32 - 23 more sweet cookies than salty cookies.
______

Ed had 29 more marbles than Doug. Ed lost 17 of his marbles at the playground. How many more marbles did Ed have than Doug then?

*mini-math-solver:*

**Answer:** ( 29.0 - 17.0 )

**Explanation:** Ed had 29 more marbles than Doug. So Doug had 29 less marbles than Ed. Ed lost 17 marbles. So Ed had 29 - 17

______

There are 270 students in a school. If the school has 5 students in each grades and each grade has the same number of students How many grades are there in total?

*mini-math-solver:*

**Answer:** ( 270.0 / 5.0 )

**Explanation:** There are 270 students. Each grade has the same number of students. So 270 / 5 students.

______

### Installation

#### macOS or Linux

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install torch numpy transformers datasets sentencepiece protobuf
```

### Usage

```shell
$ python train.py
```
