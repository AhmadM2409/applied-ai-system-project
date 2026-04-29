```mermaid
flowchart TD
    A[User Input / Sample Post] --> B[Preprocessing]
    B --> C[Rule-Based MoodAnalyzer]
    B --> D[ML Pipeline: CountVectorizer + Logistic Regression]

    E[dataset.py: SAMPLE_POSTS + TRUE_LABELS] --> C
    E --> D
    E --> F[evaluator.py]

    C --> G[Rule-Based Prediction]
    C --> H[Rule-Based Confidence Score]
    D --> I[ML Prediction]
    D --> J[ML Confidence Score]

    G --> K{Low Confidence?}
    H --> K
    K -->|Yes| L[Guardrail Output: uncertain]
    K -->|No| M[Final Rule-Based Label]

    I --> N[Final ML Label]
    J --> O[ML Confidence Output]

    F --> P[Accuracy + Average Confidence]
    F --> Q[Prediction Logs]

    P --> R[Human Review + Model Card Reflection]
    Q --> R
    L --> S[Final Output]
    M --> S
    N --> S
    O --> S
```

System Overview:

User input is processed and passed to both the rule-based and ML systems.
The rule-based system uses word matching and generates a confidence score.
The ML system uses CountVectorizer and Logistic Regression for prediction and confidence.
Guardrails return "uncertain" when rule-based confidence is too low.
evaluator.py compares predictions with true labels and computes metrics.
Human review is used for reflection and model evaluation.
