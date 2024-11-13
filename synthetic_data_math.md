Certainly! Below are all the mathematical formulas I used in the code to implement the synthetic data generation process with controlled leakage, as per your description. I've broken down each step to make it clear how the code corresponds to the mathematical expressions.

---

### **1. Feature Generation (\( \boldsymbol{X} \))**

We generate the feature matrix \( \boldsymbol{X} \in \mathbb{R}^{n \times d} \) by sampling \( n \) observations from a multivariate normal distribution:

\[
\boldsymbol{x}\_i \sim \mathcal{N}\left(\boldsymbol{\mu}\_x, \boldsymbol{\Sigma}\_x\right), \quad i = 1, \ldots, n
\]

- **\( \boldsymbol{\mu}\_x \in \mathbb{R}^d \)**: Mean vector of the features.
- **\( \boldsymbol{\Sigma}\_x \in \mathbb{R}^{d \times d} \)**: Covariance matrix of the features.

---

### **2. Ground Truth Concepts (\( \boldsymbol{c}\_i \))**

We construct the ground truth concepts \( \boldsymbol{c}\_i \in \{0,1\}^k \) as follows:

#### **2.1. Constructing Matrix \( \mathbf{A} \)**

Matrix \( \mathbf{A} \in \mathbb{R}^{k \times d} \) is designed to project the first \( b \) features into the concept space:

\[
\mathbf{A} = \left[\mathbf{R}_A \mid \mathbf{0}_{k \times (d - b)}\right]
\]

- **\( \mathbf{R}\_A \in \mathbb{R}^{k \times b} \)**: Random projection matrix with entries:

  \[
  \left(\mathbf{R}_A\right)_{jp} \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1), \quad j = 1, \ldots, k; \quad p = 1, \ldots, b
  \]

- **\( \mathbf{0}\_{k \times (d - b)} \)**: Zero matrix to exclude the remaining features.

#### **2.2. Computing Success Probabilities \( \boldsymbol{\pi}\_i \)**

We compute the logits for the ground truth concepts:

\[
\boldsymbol{\eta}\_i = \mathbf{A} \boldsymbol{x}\_i + \boldsymbol{\epsilon}\_c
\]

- **\( \boldsymbol{\epsilon}\_c \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{\Sigma}\_c\right) \)**: Noise vector with covariance \( \boldsymbol{\Sigma}\_c \in \mathbb{R}^{k \times k} \).

The success probabilities are then obtained via the sigmoid function:

\[
\boldsymbol{\pi}\_i = \sigma\left(\boldsymbol{\eta}\_i\right)
\]

- **Sigmoid function** \( \sigma(z) \):

  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

#### **2.3. Sampling Ground Truth Concepts \( \boldsymbol{c}\_i \)**

Each concept \( c\_{ij} \) is sampled from a Bernoulli distribution:

\[
c*{ij} \sim \operatorname{Bernoulli}\left(\pi*{ij}\right), \quad j = 1, \ldots, k
\]

---

### **3. Leakage Term (\( \boldsymbol{l}\_i \))**

We construct the leakage term \( \boldsymbol{l}\_i \in \mathbb{R}^k \) as follows:

#### **3.1. Constructing Matrix \( \mathbf{B} \)**

Matrix \( \mathbf{B} \in \mathbb{R}^{k \times d} \) projects selected features into the concept space:

\[
\mathbf{B} = \left[\mathbf{0}_{k \times b} \mid \mathbf{R}_B \mid \mathbf{0}_{k \times l}\right]
\]

- **\( \mathbf{0}\_{k \times b} \)**: Zero matrix to exclude the first \( b \) features.
- **\( \mathbf{R}\_B \in \mathbb{R}^{k \times (d - b - l)} \)**: Random projection matrix with entries:

  \[
  \left(\mathbf{R}_B\right)_{jq} \stackrel{\text{iid}}{\sim} \mathcal{N}(0, 1), \quad j = 1, \ldots, k; \quad q = 1, \ldots, d - b - l
  \]

- **\( \mathbf{0}\_{k \times l} \)**: Zero matrix to exclude the last \( l \) features.

#### **3.2. Computing Leakage Term \( \boldsymbol{l}\_i \)**

\[
\boldsymbol{l}\_i = \mathbf{B} \boldsymbol{x}\_i
\]

---

### **4. Estimated Concepts (\( \hat{\boldsymbol{c}}\_i \))**

We compute the estimated concepts \( \hat{\boldsymbol{c}}\_i \in [0,1]^k \) as:

\[
\hat{\boldsymbol{c}}_i = \sigma\left(\mathbf{A} \boldsymbol{x}\_i + \boldsymbol{l}\_i + \boldsymbol{\epsilon}_{\hat{c}}\right)
\]

- **\( \boldsymbol{\epsilon}_{\hat{c}} \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{\Sigma}_{\hat{c}}\right) \)**: Noise vector with covariance \( \boldsymbol{\Sigma}\_{\hat{c}} \in \mathbb{R}^{k \times k} \).
- **\( \sigma \)**: Sigmoid function as defined earlier.

---

### **5. Target Labels (\( y_i \))**

We generate the target labels \( y_i \in \{1, \ldots, J\} \) as follows:

#### **5.1. Defining Function \( f \)**

If no custom function \( f \) is provided, we define \( f: \mathbb{R}^k \times \mathbb{R}^k \rightarrow \mathbb{R}^J \) using a simple Multi-Layer Perceptron (MLP):

- **Inputs**: Concatenate \( \boldsymbol{c}\_i \) and \( \boldsymbol{l}\_i \):

  \[
  \boldsymbol{u}\_i = \left[\boldsymbol{c}_i; \boldsymbol{l}_i\right] \in \mathbb{R}^{2k}
  \]

- **First Layer**:

  \[
  \boldsymbol{h}\_i = \phi\left(\mathbf{W}\_1 \boldsymbol{u}\_i + \boldsymbol{b}\_1\right) \in \mathbb{R}^{h}
  \]

  - **\( \mathbf{W}\_1 \in \mathbb{R}^{h \times 2k} \)**: Weight matrix with entries \( \sim \mathcal{N}(0,1) \).
  - **\( \boldsymbol{b}\_1 \in \mathbb{R}^h \)**: Bias vector initialized to zeros.
  - **\( \phi(z) \)**: ReLU activation function:

    \[
    \phi(z) = \max(0, z)
    \]

- **Output Layer**:

  \[
  \boldsymbol{o}\_i = \mathbf{W}\_2 \boldsymbol{h}\_i + \boldsymbol{b}\_2 \in \mathbb{R}^{J}
  \]

  - **\( \mathbf{W}\_2 \in \mathbb{R}^{J \times h} \)**: Weight matrix with entries \( \sim \mathcal{N}(0,1) \).
  - **\( \boldsymbol{b}\_2 \in \mathbb{R}^J \)**: Bias vector initialized to zeros.

#### **5.2. Computing Target Probabilities \( \boldsymbol{p}\_i \)**

We compute the logits for the target probabilities:

\[
\boldsymbol{z}\_i = f\left(\boldsymbol{c}\_i, \boldsymbol{l}\_i\right) + \boldsymbol{\epsilon}\_y
\]

- **\( \boldsymbol{\epsilon}\_y \sim \mathcal{N}\left(\mathbf{0}, \boldsymbol{\Sigma}\_y\right) \)**: Noise vector with covariance \( \boldsymbol{\Sigma}\_y \in \mathbb{R}^{J \times J} \).

We then apply the softmax function to obtain the probabilities:

\[
\boldsymbol{p}\_i = \operatorname{softmax}\left(\boldsymbol{z}\_i\right)
\]

- **Softmax function**:

  \[
  \operatorname{softmax}\left(\boldsymbol{z}_i\right)\_j = \frac{\exp\left(z_{ij}\right)}{\sum*{k=1}^{J} \exp\left(z*{ik}\right)}, \quad j = 1, \ldots, J
  \]

#### **5.3. Sampling Target Labels \( y_i \)**

We sample \( y_i \) from a categorical distribution based on \( \boldsymbol{p}\_i \):

\[
y_i \sim \operatorname{Categorical}\left(\boldsymbol{p}\_i\right)
\]

- This results in \( y_i \in \{1, 2, \ldots, J\} \).

---

### **6. Constraints**

To ensure the random projections and leakage control function properly, the following constraints must be satisfied:

\[
k < b < d - k - l
\]

- **\( k \)**: Number of concepts.
- **\( b \)**: Number of features used in ground truth concepts.
- **\( d \)**: Total number of features.
- **\( l \)**: Number of features excluded from leakage.

---

### **7. Summary of Variables and Parameters**

- **\( n \)**: Number of observations.
- **\( d \)**: Dimensionality of features.
- **\( k \)**: Number of concepts.
- **\( J \)**: Number of target classes.
- **\( b \)**: Number of features used in ground truth concepts.
- **\( l \)**: Number of features excluded from leakage.
- **\( \boldsymbol{\mu}\_x \in \mathbb{R}^d \)**: Mean vector of features.
- **\( \boldsymbol{\Sigma}\_x \in \mathbb{R}^{d \times d} \)**: Covariance matrix of features.
- **\( \boldsymbol{\Sigma}\_c \in \mathbb{R}^{k \times k} \)**: Covariance matrix of noise in ground truth concepts.
- **\( \boldsymbol{\Sigma}\_{\hat{c}} \in \mathbb{R}^{k \times k} \)**: Covariance matrix of noise in estimated concepts.
- **\( \boldsymbol{\Sigma}\_y \in \mathbb{R}^{J \times J} \)**: Covariance matrix of noise in target logits.
- **\( f \)**: Function mapping concepts and leakage to target logits.

---

### **8. Relationship Between Code and Formulas**

- **Feature Generation**:

  ```python
  X = np.random.multivariate_normal(mu_x, Sigma_x, size=n)
  ```

  Corresponds to \( \boldsymbol{x}\_i \sim \mathcal{N}\left(\boldsymbol{\mu}\_x, \boldsymbol{\Sigma}\_x\right) \).

- **Constructing \( \mathbf{A} \) and \( \mathbf{B} \)**:

  ```python
  R_A = np.random.normal(0, 1, size=(k, b))
  A = np.hstack([R_A, np.zeros((k, d - b))])
  ```

  ```python
  R_B = np.random.normal(0, 1, size=(k, d - b - l))
  B = np.zeros((k, d))
  B[:, b : d - l] = R_B
  ```

  Corresponds to the constructions of \( \mathbf{A} \) and \( \mathbf{B} \) as per the formulas.

- **Computing \( \boldsymbol{\pi}\_i \) and Sampling \( \boldsymbol{c}\_i \)**:

  ```python
  epsilon_c = np.random.multivariate_normal(np.zeros(k), Sigma_c, size=n)
  logits_pi = np.dot(X, A.T) + epsilon_c
  pi_i = sigmoid(logits_pi)
  c = np.random.binomial(1, pi_i)
  ```

  Corresponds to:

  \[
  \boldsymbol{\pi}\_i = \sigma\left(\mathbf{A} \boldsymbol{x}\_i + \boldsymbol{\epsilon}\_c\right)
  \]

  \[
  c*{ij} \sim \operatorname{Bernoulli}\left(\pi*{ij}\right)
  \]

- **Computing Leakage Term \( \boldsymbol{l}\_i \)**:

  ```python
  l = np.dot(X, B.T)
  ```

  Corresponds to \( \boldsymbol{l}\_i = \mathbf{B} \boldsymbol{x}\_i \).

- **Computing Estimated Concepts \( \hat{\boldsymbol{c}}\_i \)**:

  ```python
  epsilon_hat_c = np.random.multivariate_normal(np.zeros(k), Sigma_hat_c, size=n)
  logits_hat_c = np.dot(X, A.T) + l + epsilon_hat_c
  hat_c = sigmoid(logits_hat_c)
  ```

  Corresponds to \( \hat{\boldsymbol{c}}_i = \sigma\left(\mathbf{A} \boldsymbol{x}\_i + \boldsymbol{l}\_i + \boldsymbol{\epsilon}_{\hat{c}}\right) \).

- **Computing Target Labels \( y_i \)**:

  - **When \( f \) is not provided**:

    ```python
    def f(c_i, l_i):
        inputs = np.hstack([c_i, l_i])
        hidden = np.maximum(0, np.dot(inputs, W1) + b1)
        outputs = np.dot(hidden, W2) + b2
        return outputs
    ```

    Corresponds to the MLP:

    \[
    \boldsymbol{u}\_i = \left[\boldsymbol{c}_i; \boldsymbol{l}_i\right]
    \]

    \[
    \boldsymbol{h}\_i = \phi\left(\mathbf{W}\_1 \boldsymbol{u}\_i + \boldsymbol{b}\_1\right)
    \]

    \[
    \boldsymbol{o}\_i = \mathbf{W}\_2 \boldsymbol{h}\_i + \boldsymbol{b}\_2
    \]

  - **Computing \( \boldsymbol{p}\_i \) and Sampling \( y_i \)**:

    ```python
    epsilon_y = np.random.multivariate_normal(np.zeros(J), Sigma_y, size=n)
    logits_p = f(c, l) + epsilon_y
    p_i = softmax(logits_p)
    y = np.array([np.random.choice(J, p=p_i[i]) for i in range(n)]) + 1
    ```

    Corresponds to:

    \[
    \boldsymbol{p}\_i = \operatorname{softmax}\left(f\left(\boldsymbol{c}\_i, \boldsymbol{l}\_i\right) + \boldsymbol{\epsilon}\_y\right)
    \]

    \[
    y_i \sim \operatorname{Categorical}\left(\boldsymbol{p}\_i\right)
    \]

---

### **9. Functions Used**

- **Sigmoid Function**:

  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- **ReLU Activation Function**:

  \[
  \phi(z) = \max(0, z)
  \]

- **Softmax Function**:

  \[
  \operatorname{softmax}\left(\boldsymbol{z}\right)_j = \frac{\exp\left(z_j\right)}{\sum_{k=1}^{J} \exp\left(z_k\right)}
  \]

---

### **10. Random Projections and the Johnson-Lindenstrauss Lemma**

By using random matrices \( \mathbf{R}\_A \) and \( \mathbf{R}\_B \) with entries sampled from \( \mathcal{N}(0,1) \), we ensure that the projections approximately preserve distances between high-dimensional feature vectors when mapped into lower-dimensional concept spaces, as per the Johnson-Lindenstrauss lemma.

- **Constraints for Random Projections**:

  - For \( \mathbf{A} \):

    \[
    k < b
    \]

  - For \( \mathbf{B} \):

    \[
    k < d - b - l
    \]

  - Combined Constraint:

    \[
    k < b < d - k - l
    \]

---

### **11. Noise Terms**

- **\( \boldsymbol{\epsilon}\_c \)**: Noise in ground truth concepts.
- **\( \boldsymbol{\epsilon}\_{\hat{c}} \)**: Noise in estimated concepts.
- **\( \boldsymbol{\epsilon}\_y \)**: Noise in target logits.

Each noise term is sampled from a multivariate normal distribution with zero mean and specified covariance matrices.

---

### **12. Custom Function \( f \)**

If a custom function \( f \) is provided, it should map the concepts and leakage to target logits:

\[
f: \mathbb{R}^k \times \mathbb{R}^k \rightarrow \mathbb{R}^J
\]

- The user can define \( f \) as needed for specific experiments.

---

### **13. Random Seed**

- A random seed is used to ensure replicability of the synthetic data generation process.

---

I hope this comprehensive breakdown of the mathematical formulas helps you verify that the code accurately implements your described data generation process. If you have any questions or need further clarification on any part, please let me know!
