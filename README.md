# Mini Math Solver

A small language model capable of solving math word problems

Training on a single GPU allows a 250M-parameter mini-math-solver to acquire step-by-step reasoning abilities from a larger model, enabling accurate problem-solving

*Sample problem:*

Julia played soccer with 2 friends on Monday, 14 friends on Tuesday, and 16 friends on Wednesday. How many friends did she play with on Tuesday and Wednesday? 

**mini-math-solver:**

Julia played with 14 friends on tuesday. She played with 16 friends on wednesday. So she played with 14 + 16  


### Installation

#### macOS or Linux

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install torch numpy transformers datasets sentencepiece protobuf
```
