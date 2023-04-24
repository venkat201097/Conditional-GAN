<h1> Generative Adversarial Networks </h1>

This is my implementation (from scratch) of GAN on the MNIST dataset. The architecture is a simple MLP with LeakyReLU activation and discriminator dropout. I implemented the conditional version of GAN (Mirza and Osindero : https://arxiv.org/pdf/1411.1784.pdf) by conditioning on the MNIST digit labels. The generator models the conditional data distribution $P(X|MNIST digit label)$.

### Conditional GAN training (digit-wise outputs)

<p align="center">
    <img src="Training.gif" width="80%">
</p>

### Branches
1. main : Has the implementation of conditional GAN.
2. vanilla-GAN : GAN for modeling unconditional data distribution $P(X).