# penr-oz-neural-network-v2-torch-nn
Version 2 Implementation of Neural Network microservice leveraging 
PyTorch library's neural network (nn) package

This repository demonstrates same key concepts in neural networks as in [penr-oz-neural-network-torch](https://github.com/derinworks/penr-oz-neural-network-torch) 
with automatic gradient descent calculations relying on PyTorch library and 
slight changes to API signature to specifically align with leveraging Neural Network (nn) package.

Implementation follows:
* [nn-zero-to-hero lectures](https://github.com/karpathy/nn-zero-to-hero)
* [makemore](https://github.com/karpathy/makemore)
* [nanoGPT](https://github.com/karpathy/nanoGPT)

### Backpropagation: Auto Gradient Calculation

The gradients are automatically computed using [PyTorch](https://github.com/pytorch/pytorch) and 
the [PyTorch Neural Network package](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial)

## Quickstart Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/derinworks/penr-oz-neural-network-v2-torch-nn.git
   cd penr-oz-neural-network-v2-torch-nn
   ```

2. **Create and Activate a Virtual Environment**:
   - **Create**:
     ```bash
     python -m venv venv
     ```
   - **Activate**:
     - On Unix or macOS:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Service**:
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload
   ```

5. **Interact with the Service**
Test the endpoints using Swagger at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

6. **Interact with the Dashboard**
Diagnose model training at [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard).

---

## Testing and Coverage

To ensure code quality and maintainability, follow these steps to run tests and check code coverage:

1. **Run Tests with Coverage**:
   Execute the following commands to run tests and generate a coverage report:
   ```bash
   coverage run -m pytest
   coverage report
   ```

2. **Generate HTML Coverage Report** (Optional):
   For a detailed coverage report in HTML format:
   ```bash
   coverage html
   ```
   Open the `htmlcov/index.html` file in a web browser to view the report.
