# Senjyu

Senjyu is a framework for parallel machine learning using MPI.

## supported algorithms

| algorithm     | example                                  |
| ------------- | ---------------------------------------- |
| Random Forest | [example](examples/test_randomforest.py) |
| K-means       | Coming Soon                              |

## example

You can easily experiment paralell random forest on iris dataset.

    mpiexec -np 3 python examples/test_randomforest.py