# numpy_autograd
In this repo I aim to motivate and show how to write an automatic differentiation library. There are various strategies to perform automatic differentiation and they each have different strengths and weaknesses. For a an overview of various methods used please refer to [1]. Py-Torch uses a graph based automatic differentiation.

Every operation performed on tensors can be shown as a DAG (directed acylic graph). In the case of neural networks, the loss value calculated for a given mini-batch is the last node of the graph. Chain rule is very powerful and yet a very simple rule. Thinking in terms of the DAG, what chain rule tells us to take the derivative on a node if the output gradient of the node is completely accumulated. If we somehow make each node in this graph to remember its parents. We can run a topological sort on the DAG and call the derivative function of the nodes in this order. That's a very simple overview of how auto-grad in [PyTorch](https://pytorch.org/) works and it is very simple to implement! Let's do it.

[1] Automatic differentiation in machine learning: a survey https://arxiv.org/abs/1502.05767
