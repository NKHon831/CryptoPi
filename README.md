<img src="assets/cryptopi_logo.png" alt="CryptoPi Logo" width="100"/>

# CryptoPi: Built for Crypto. Powered by Intelligence.

CryptoPi is a proposed Python library tailored for the **UM Hackathon 2025 – Balaena Quant Challenge (Domain 2)**. Our solution directly addresses the challenge of building robust alpha-generating trading strategies using Machine Learning (ML) on **on-chain crypto data**.
[Click here to view our pitch deck.](https://www.canva.com/design/DAGkVzm-wM8/OcP3ndDx7Df2SLinLGI8Ig/view?utm_content=DAGkVzm-wM8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h6ab8f98cd7)

## Problem Statement

Traditional backtesting frameworks fall short in the evolving landscape of crypto trading, constraining alpha generation.

### Pain Points 🚨

1. **Lack of ML Integration**: Most existing backtesting frameworks are not designed with machine learning in mind. Integrating ML pipelines—such as sentiment models, Hidden Markov Models (HMMs), or transformers—requires cumbersome custom code. This lack of native support discourages data-driven strategies and limits the potential for intelligent signal generation.
2. **Slow Performance on Large Datasets**: Frameworks like Backtrader, while powerful, are often bottlenecked by Python’s single-threaded performance. Running simulations on high-frequency or long-term historical datasets can become painfully slow, especially when testing multiple strategies or models in parallel.
3. **Lack of Experiment Tracking and Reproducibility**: Backtest results are difficult to reproduce without manual logging and versioning. There is no standardized way to track which data, models, or parameters were used, making it hard to validate findings or roll back to previous experiments. This severely impacts research transparency and credibility.

## Proposed Solution

We propose a model-centric backtesting library built for the crypto domain, creating data-driven, adaptive trading strategies that better reflect the complexities of the crypto market.

### Unique Selling Points 🚀

1. **ML-Native Design**: CryptoPi natively supports the seamless integration of machine learning models, including sentiment analysis, HMMs, and deep learning architectures. This allows researchers and quant developers to focus on building data-driven strategies instead of wrestling with incompatible toolchains. Our modular design encourages experimentation, enabling strategies to be discovered rather than hardcoded.
2. **Smart Caching Layer**: CryptoPi implements a smart caching mechanism that stores intermediate results such as preprocessed datasets, extracted features, and model outputs. This dramatically reduces redundant computation, allowing faster experimentation by reusing previously computed results for similar runs or parameter sets.
3. **Versioning & Reproducibility**: Every backtest run in CryptoPi is automatically versioned. This includes metadata such as dataset versions, model parameters, strategy configurations, and performance metrics. It ensures that experiments are transparent, traceable, and fully reproducible—crucial for maintaining research integrity and enabling enterprise-grade auditability.

## Prototype

Please refer to our pitching deck [here](https://www.canva.com/design/DAGkVzm-wM8/OcP3ndDx7Df2SLinLGI8Ig/view?utm_content=DAGkVzm-wM8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h6ab8f98cd7) to view the detailed prototype design illustrated through different diagrams and figures.

## Impacts

1. ⚡ **Accelerated Strategy Development**: By streamlining the integration of machine learning and automating key components of the backtesting process, CryptoPi helps users avoid the traditional trial-and-error approach to strategy design. This reduces development time while encouraging creativity and innovation in discovering effective trading strategies.
2. 🌍 **Increased Accessibility**: CryptoPi lowers the barrier to entry for algorithmic trading. With intuitive visualizations, a modular plug-and-play design, and support for pretrained models, even users without deep coding expertise can explore, test, and understand sophisticated strategies.
3. 📊 **Data-Driven Decision Making**: CryptoPi promotes confidence through data. By providing transparent and reproducible metrics on strategy performance, users can rely on clear, quantitative insights instead of gut feelings—ultimately reducing emotional or impulsive trading decisions.

## Future Improvements

1. 🟢 **Live Trading & Real-Time Simulation**: Integrate real-time market data and execute strategies live, with support for simulation in fast-forward mode. This will bridge the gap between research and deployment.
2. 🎛️ **Visual Strategy Builder**: A drag-and-drop UI for building, editing, and visualizing trading logic—no coding required. This will democratize strategy creation for non-technical users and accelerate rapid prototyping for technical ones.
3. 🤖 **AutoML Pipeline with Online Learning**: Automatically select, train, and fine-tune models using live feedback from market data. This will enable adaptive strategies that learn and evolve in real time, reducing the need for constant manual retraining.
4. 🧠 **Reinforcement Learning Strategies**: Introduce reinforcement learning agents that can learn optimal trading policies through trial-and-error in simulated or real environments. This will unlock next-generation strategies that go beyond predefined rules or supervised learning.

## Our Team: **6thSense**

| Name               | Role             |
| ------------------ | ---------------- |
| Ng Khai Hon        | Project Manager  |
| Chong Boon Ping    | ML Engineer      |
| Tneoh Chuan Lin    | Backend Engineer |
| Poh Sharon         | Data Engineer    |
| Vanessa Jing Taing | Product Manager  |
