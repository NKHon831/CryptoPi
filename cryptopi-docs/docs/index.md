# CryptoPi: Built for Crypto. Powered by Intelligence.

CryptoPi is a Python library tailored for the **UM Hackathon 2025 – Balaena Quant Challenge (Domain 2)**. Our solution directly addresses the challenge of building robust alpha-generating trading strategies using Machine Learning (ML) on **on-chain crypto data**.

Explore the full documentation [here](https://nkhon831.github.io/CryptoPi/) to get started.

## Problem Statement

Traditional backtesting frameworks fall short in the evolving landscape of crypto trading, constraining alpha generation.

### Pain Points 🚨

1. **Lack of ML Integration**: Most existing backtesting frameworks are not designed with machine learning in mind. Integrating ML pipelines—such as sentiment models, Hidden semi-Markov Models (HSMMs), or transformers—requires cumbersome custom code. This lack of native support discourages data-driven strategies and limits the potential for intelligent signal generation.
2. **Slow Performance on Large Datasets**: Frameworks like Backtrader, while powerful, are often bottlenecked by Python’s single-threaded performance. Running simulations on high-frequency or long-term historical datasets can become painfully slow, especially when testing multiple strategies or models in parallel.
3. **Lack of Experiment Tracking and Reproducibility**: Backtest results are difficult to reproduce without manual logging and versioning. There is no standardized way to track which data, models, or parameters were used, making it hard to validate findings or roll back to previous experiments. This severely impacts research transparency and credibility.

## Proposed Solution

We propose a model-centric backtesting library built for the crypto domain, creating data-driven, adaptive trading strategies that better reflect the complexities of the crypto market.

### Unique Selling Points 🚀

1. **ML-Native Design**: CryptoPi natively supports the seamless integration of machine learning models, including sentiment analysis, HMMs, and deep learning architectures. This allows researchers and quant developers to focus on building data-driven strategies instead of wrestling with incompatible toolchains. Our modular design encourages experimentation, enabling strategies to be discovered rather than hardcoded.
2. **Smart Caching Layer**: CryptoPi implements a smart caching mechanism that stores intermediate results such as preprocessed datasets, extracted features, and model outputs. This dramatically reduces redundant computation, allowing faster experimentation by reusing previously computed results for similar runs or parameter sets.
3. **Versioning & Reproducibility**: Every backtest run in CryptoPi is automatically versioned. This includes metadata such as dataset versions, model parameters, strategy configurations, and performance metrics. It ensures that experiments are transparent, traceable, and fully reproducible—crucial for maintaining research integrity and enabling enterprise-grade auditability.

<!-- ## Prototype

Please refer to our pitching deck [here](https://www.canva.com/design/DAGkVzm-wM8/OcP3ndDx7Df2SLinLGI8Ig/view?utm_content=DAGkVzm-wM8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h6ab8f98cd7) for more detailed prototype design illustrated through different diagrams and figures. You can also refer to our Lucidchart [here](https://lucid.app/lucidchart/95860ad1-6bf8-45e8-9e23-ed220dede91e/edit?viewport_loc=5154%2C-1194%2C3409%2C1417%2Cf~U95iUYWTq-&invitationId=inv_282566a2-a946-4eab-b096-b330419bc219) for the prototype diagrams. -->

<!-- ### Conceptual Architecture Diagram:

<img src="assets\conceptual_architecture_diagram.png" alt="Conceptual Architecture Diagram" width="900"/> -->

## Model Selections

1. **Regime Detection Model - Hidden Markov Model (HMM)**: The HMM was selected for its strengths in modeling temporal patterns in market behavior:

- 🔍 Captures Hidden Market States: Effectively identifies latent regimes such as bull, bear, or sideways trends.
- 🎲 Probabilistic Transitions: Models regime shifts with realistic probabilities, useful for dynamic environments.
- 🪶 Lightweight & Efficient: Offers a favorable trade-off between computational cost and accuracy, suitable for scalable or resource-constrained pipelines.

2. **Alpha Model - Logistic Regression**: Logistic Regression was chosen for its balance of simplicity, speed, and reliability:

- 🔐 Robust with Limited Data: Performs well even with smaller datasets, avoiding overfitting.
- 📊 Probabilistic Outputs: Generates interpretable signal confidence levels to guide position sizing.
- 🔍 Interpretability: Clear feature importance makes it easier to audit and refine strategies.
- ⚡ Fast Inference Time: Low latency makes it ideal for real-time trade signal generation.

## Impacts

1. ⚡ **Accelerated Strategy Development**: By streamlining the integration of machine learning and automating key components of the backtesting process, CryptoPi helps users avoid the traditional trial-and-error approach to strategy design. This reduces development time while encouraging creativity and innovation in discovering effective trading strategies.
2. 🌍 **Increased Accessibility**: CryptoPi lowers the barrier to entry for algorithmic trading. With intuitive visualizations, a modular plug-and-play design, and support for pretrained models, even users without deep coding expertise can explore, test, and understand sophisticated strategies.
3. 📊 **Data-Driven Decision Making**: CryptoPi promotes confidence through data. By providing transparent and reproducible metrics on strategy performance, users can rely on clear, quantitative insights instead of gut feelings—ultimately reducing emotional or impulsive trading decisions.

## Model Performance

1. **Sharpe Ratio (SR)** >= 1.8

   Indicates strong risk-adjusted returns, suggesting the strategy generates significantly more return per unit of volatility than a passive benchmark.

2. **Max Drawdown (MDD)** > -30%

   Maintains capital preservation with acceptable downside risk, ensuring the strategy avoids catastrophic losses during adverse market conditions.

3. **Trade Frequency** >= 3%

   Balances between overtrading and undertrading by maintaining a healthy level of trading activity, enabling sufficient market participation without excessive noise.

## Future Improvements

1. 🟢 **Live Trading & Real-Time Simulation**: Integrate real-time market data and execute strategies live, with support for simulation in fast-forward mode. This will bridge the gap between research and deployment.
2. ⚙️ **Model Enhancement & Finetuning**: Upgrade models to more advanced architectures and apply dynamic finetuning strategies for improved signal accuracy and robustness.
3. 🧠 **Multi-Modal Fusion**: Combine sentiment, technical, fundamental, and on-chain data using advanced fusion techniques to capture richer market insights.
4. 📊 **Data & Feature Engineering**: Engineer more alpha-rich features and apply real-time pipelines to adapt quickly to changing market conditions.

## Our Team: **6thSense**

| Name               | Role             |
| ------------------ | ---------------- |
| Ng Khai Hon        | Project Manager  |
| Chong Boon Ping    | ML Engineer      |
| Tneoh Chuan Lin    | Backend Engineer |
| Poh Sharon         | Data Engineer    |
| Vanessa Jing Taing | Product Manager  |
