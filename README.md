# Credit Risk Scoring Model – Bati Bank & Xente

## Credit Scoring Business Understanding

### Basel II Accord and the Demand for Interpretable, Well‑Documented Models

The Basel II Capital Accord requires financial institutions to quantify credit risk with a high degree of transparency and auditability. Under the Internal Ratings‑Based (IRB) approach, banks must:

- Categorize borrowers into risk buckets with meaningful probability of default (PD) estimates.
- Demonstrate that their rating system is **conceptually sound** and has **predictive power**.
- Maintain thorough documentation of model design, variable selection, and validation.

These regulatory requirements mean that a “black‑box” model is difficult to justify. Even when high‑performance ensemble methods are used, banks must be able to explain **how** features influence the score, **why** a particular threshold was chosen, and **what** the model’s limitations are. Interpretability and documentation are therefore not optional niceties – they are regulatory necessities.

### Proxy Variable: Why It Is Necessary and the Business Risks It Introduces

The raw transaction dataset contains **no default label** – there is no column indicating whether a customer failed to repay a loan. To build a credit scoring model, we must engineer a **proxy target** that approximates a credit default event. We define this proxy using **RFM (Recency, Frequency, Monetary) segmentation**: customers who are disengaged (high recency, low frequency, low monetary value) are labeled as “high‑risk” (likely to default), while active, high‑value customers are labeled “low‑risk.”

**Why a proxy is necessary**  
Without a direct default flag, we cannot apply supervised learning to predict actual defaults. The proxy transforms behavioral patterns into a risk indicator, allowing the model to learn a mapping from transaction features to a “risk score.” This is common in alternative credit scoring, where traditional credit bureau data is unavailable.

**Business risks introduced by proxy‑based prediction**  

- **Misclassification:** Some disengaged customers may be falsely labeled as high‑risk even though they would have repaid a loan. Conversely, active customers could still default.
- **Self‑fulfilling prophecy:** If the model is used to deny credit to “high‑risk” labeled customers, we may never observe whether they would have actually defaulted, reinforcing the proxy’s bias.
- **Regulatory scrutiny:** A proxy‑derived target must be defended to auditors. The bank must clearly state that it is a **modeling assumption**, not an observed outcome.
- **Drift:** Customer behavior changes over time; the proxy definition must be periodically reviewed and recalibrated.

### Trade‑offs: Interpretable Models vs. High‑Performance Models

In a regulated financial context, the choice of model involves a trade‑off between **interpretability** and **predictive performance**.

| Aspect | Interpretable Model (e.g., Logistic Regression with Weight‑of‑Evidence transformation) | High‑Performance Model (e.g., Gradient Boosting, XGBoost) |
|--------|-----------------------------------------------|------------------------------------------------------------|
| **Explainability** | Coefficients directly show the direction and magnitude of each feature’s impact. Business stakeholders can easily understand how a score is derived. | Feature importance can be calculated (e.g., SHAP values), but the interactions between features are complex and harder to communicate. |
| **Regulatory acceptance** | Highly favored by regulators and internal audit. WoE‑based models are standard in credit scoring because they enforce monotonic relationships between features and risk. | Must be accompanied by additional interpretability tools and documented justification. Regulators may require a supplementary interpretable model for validation. |
| **Performance** | May underperform on non‑linear patterns and interactions. However, with well‑engineered features, logistic regression can be competitive. | Often achieves higher accuracy, especially with large, high‑dimensional datasets. |
| **Stability** | Generally more stable and less prone to overfitting if feature engineering is sound. | Can overfit if not carefully regularized and tuned. Requires more rigorous monitoring. |
| **Implementation** | Easier to deploy and score in real time; can be expressed as a simple scorecard. | More complex to deploy (serialized model artifacts) and requires more infrastructure. |

**Our approach:** We will first build a **Logistic Regression model with Weight‑of‑Evidence features** to serve as a transparent, regulator‑ready baseline. We will then experiment with **Gradient Boosting (XGBoost)** to assess the performance gain. The final model selection will be justified against both the regulatory need for interpretability and the business need for accurate risk discrimination. Experiment tracking (MLflow) and rigorous documentation will be maintained throughout.