

```markdown
# Personalized Customer Retention using Causal Uplift Modeling, CatBoost X-Learner & Generative AI RAG

# AI-Powered Customer Retention Decision Support System

## Overview

This project focuses on **customer retention using Causal Uplift Modeling combined with Generative AI**.

Unlike traditional churn prediction systems that only identify customers likely to leave, this system predicts:

> **Which customers are most likely to respond positively to a retention intervention and generate maximum business value.**

The project combines:

- **CatBoost-based X-Learner Uplift Modeling**
- **SHAP Explainable AI**
- **ROI-driven campaign optimization**
- **Customer segmentation**
- **Gemini Generative AI**
- **Retrieval-Augmented Generation (RAG)**

The final solution is deployed using **Streamlit** as an interactive AI-powered customer retention analytics dashboard.

---

# Business Problem

Traditional churn prediction answers:

> "Who is likely to churn?"

However, businesses need to answer:

> "Who should receive a retention offer to maximize ROI?"

Sending discounts or incentives to every risky customer increases marketing costs.

This project solves this challenge using **Causal Uplift Modeling**, which estimates the incremental impact of a treatment and identifies customers who will change their behavior because of an intervention.

The system further enhances decision-making using **Generative AI**, allowing business users to ask questions and receive recommendations grounded in company documents.

---

# Dataset

## IBM Telco Customer Churn Dataset

The dataset contains:

- Customer demographics
- Subscription information
- Service usage
- Billing details
- Contract information
- Payment methods
- Churn outcomes

---

# Complete AI System Architecture

```

Customer Data
|
▼
Data Preprocessing
|
▼
CatBoost Uplift Model
|
▼
Customer Treatment Effect Prediction
|
▼
Customer Segmentation
|
▼
ROI Calculation
|
▼
SHAP Explainability
|
▼
Gemini AI + RAG Knowledge Assistant
|
▼
Business Recommendations
|
▼
Streamlit Dashboard

```

---

# Machine Learning Pipeline

## 1. Data Preprocessing

Performed:

- Data cleaning
- Missing value handling
- Feature engineering
- Categorical encoding
- Feature selection
- Correlation analysis

---

# 2. Treatment and Outcome Construction

Created causal uplift learning variables.

## Y0 - Control Outcome

Represents customer retention without treatment.

```

Y0 = 1 → Customer retained without treatment
Y0 = 0 → Customer churned without treatment

```

## Y1 - Treatment Outcome

Represents customer retention after receiving treatment.

```

Y1 = 1 → Customer retained after treatment
Y1 = 0 → Customer churned after treatment

```

Using these outcomes, uplift labels were generated for causal modeling.

---

# 3. Uplift Modeling

Implemented multiple meta-learning approaches:

- S-Learner
- T-Learner
- X-Learner


## Base Model

CatBoost Classifier


## Final Model

CatBoost + X-Learner


The model estimates:

```

Uplift = Probability(Y1) - Probability(Y0)

```

which represents the expected incremental impact of providing treatment.

---

# 4. Customer Segmentation

Customers are classified into four uplift groups.

## Persuadables

Customers who are likely to stay only when treatment is provided.

Recommended:

- High priority campaigns
- Retention offers


## Sure Things

Customers who stay regardless of intervention.

Recommended:

- No incentive required


## Sleeping Dogs

Customers who may react negatively to treatment.

Recommended:

- Avoid targeting


## Lost Causes

Customers unlikely to respond even after treatment.

Recommended:

- Low marketing priority

---

# 5. ROI-Based Decision Engine

The system calculates business impact using:

- Treatment cost
- Expected retained customers
- Revenue generated

Metrics calculated:

- Expected revenue
- Campaign cost
- Net profit
- ROI


Average achieved:

```

ROI ≈ ₹80 per customer

```

---

# Explainable AI using SHAP

SHAP analysis is integrated to explain uplift predictions.

The explainability layer identifies:

- Features increasing treatment effectiveness
- Customer characteristics influencing uplift
- Important drivers behind recommendations


This enables transparent decision-making for marketing teams.

---

# Generative AI Integration

## Gemini AI Business Assistant

The dashboard integrates Google's Gemini model to provide AI-powered business insights.

The AI assistant can:

- Analyze campaign performance
- Explain customer recommendations
- Generate executive summaries
- Suggest retention strategies
- Answer marketing-related questions

---

# Retrieval-Augmented Generation (RAG)

A custom RAG pipeline was implemented to provide Gemini with company-specific knowledge.

Instead of relying only on general LLM knowledge, the AI retrieves relevant information from internal business documents.

## RAG Pipeline

```

Business Documents
|
▼
Document Loader
|
▼
Text Chunking
|
▼
Sentence Transformer Embeddings
|
▼
FAISS Vector Database
|
▼
User Query
|
▼
Semantic Retrieval
|
▼
Relevant Context
|
▼
Gemini 2.5 Flash
|
▼
Business Answer

```

---

# RAG Components

## Document Processing

The system loads business knowledge documents and converts them into searchable text chunks.

---

## Embedding Generation

Uses:

```

Sentence Transformer
all-MiniLM-L6-v2

```

to convert text into numerical embeddings.

---

## Vector Search

FAISS is used for efficient similarity search.

The system retrieves the most relevant business information based on user queries.

---

## Grounded AI Responses

Gemini receives:

- User question
- Retrieved business context
- System instructions

This reduces hallucination and provides more relevant business recommendations.

---

# Streamlit AI Dashboard

The dashboard provides:

## Executive Dashboard

Displays:

- Total customers
- Average uplift
- Average ROI
- Best performing segments
- AI-generated campaign summaries


---

## Customer Explorer

Users can search customers and view:

- Customer profile
- Predicted uplift
- Customer segment
- ROI estimate
- Recommended retention action
- AI-generated customer analysis


---

## Campaign Insights

AI generates:

- Target customer recommendations
- Budget suggestions
- Expected campaign impact
- Marketing strategies


---

## Ask AI Assistant

Users can ask business questions related to:

- Customer retention
- Marketing campaigns
- Sales strategy
- CRM decisions
- Revenue optimization


The AI assistant uses RAG-based retrieval to provide context-aware responses.

---

# Inference Pipeline

```

User
|
▼
Streamlit Application
|
▼
Inference Pipeline
|
├── Load ML Model
├── Load Feature Pipeline
├── Generate Uplift Prediction
├── Customer Segmentation
├── ROI Calculation
├── SHAP Explanation
|
▼
Gemini AI + RAG
|
▼
Business Recommendation

```

---

# Technology Stack

## Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XLearner
- SHAP


## Generative AI

- Google Gemini 2.5 Flash
- Retrieval-Augmented Generation
- Sentence Transformers
- FAISS Vector Database


## Visualization & Deployment

- Streamlit
- Matplotlib
- Plotly
- Altair
- Pickle


---

# Key Results

- Developed an end-to-end causal uplift modeling pipeline.
- Compared S-Learner, T-Learner, and X-Learner approaches.
- Selected CatBoost X-Learner as the final uplift model.
- Achieved approximately **18% average uplift**.
- Estimated approximately **₹80 ROI per customer**.
- Built customer-level retention recommendations.
- Added SHAP explainability for transparent predictions.
- Integrated Gemini Generative AI for business analysis.
- Developed a RAG-based enterprise knowledge assistant using FAISS.
- Created an AI-powered Streamlit dashboard for marketing decision support.

---

# Future Improvements

## Machine Learning

- Real-time uplift prediction API
- Automated model retraining pipeline
- Multi-treatment uplift modeling
- Dynamic treatment optimization


## Generative AI

- Support PDF and DOCX knowledge sources
- Add conversational memory
- Add customer-specific RAG retrieval
- Deploy vector database using managed cloud services


## Business Intelligence

- A/B testing integration
- Campaign automation
- Real-time CRM integration
- Marketing budget optimization
```


