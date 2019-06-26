# Welcome to Sotabench

## Overview

### Mission

Build a massive resource of high quality machine learning implementations, 
giving the community baselines that they can easily reuse, extend
and build upon, and ultimately helping advance the state-of-the-art in the field.

### How it works

Every model implementation is evaluated according to public benchmarks. This allows us to test some basic 
functionality of the model: 

- Can we run the model?
- Does the model reproduce the results of the original paper?
- How long does inference take? What are the computational requirements?

If a model passes these tests, we have confidence that we can use it as a building block for successive tasks 
such as transfer learning, demoing, real-time inference and more.

### How do I get involved?

You've come to the right place! These docs contain everything you need to know about Sotabench! In the following 
section you will find instructions for installation, after which you can get started with the following:

- [Getting Started]() - Start here if you are new to Sotabench. We will walk through an example of taking a 
state-of-the-art image classification model and submitting it to Sotabench.
- [Reference Guide]() - Look here for technical references for the Sotabench API. They describe the mechanics of how the library works. Basic understanding of key concepts assumed.

## Installation

Sotabench is a Python library that contains the benchmarks, datasets and other tools to evaluate your implementations.

There are two ways to install sotabench:

**Install Sotabench from PyPi**

    pip install sotabench

**Install Sotabench from GitHub source**

    git clone https://github.com/sotabench/sotabench.git
    cd sotabench
    python setup.py install

## Support

If you get stuck you can head to our [Discourse]() forum where you ask questions on how to use the project. 
You can also find ideas for contributions, and work with others on exciting projects.
