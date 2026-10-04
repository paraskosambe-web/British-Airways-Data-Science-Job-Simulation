<!-- ═══════════════════════ HEADER ═══════════════════════ -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:01295c,55:075aaa,100:eb2226&height=290&section=header&text=British%20Airways&fontSize=68&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=Data%20Science%20Job%20Simulation%20%E2%80%A2%20Forage&descSize=22&descAlignY=60" width="100%" alt="British Airways Data Science Job Simulation"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=21&duration=3000&pause=900&color=EB2226&center=true&vCenter=true&width=820&height=50&lines=%E2%9C%88%EF%B8%8F+Predicting+who+will+complete+a+flight+booking;%F0%9F%94%8E+Explore+%E2%86%92+Engineer+%E2%86%92+Model+%E2%86%92+Validate;%F0%9F%8C%B2+Random+Forest+%2B+5-Fold+Stratified+Cross-Validation;%F0%9F%93%88+Feature+importance+and+business+insight" alt="Typing animation"/>

<br/>

<!-- Animated flight path (file: assets/flight-path.svg) -->
<img src="./assets/flight-path.svg" width="90%" alt="Animated flight path from flight search to completed booking"/>

<br/><br/>

<img src="https://img.shields.io/badge/Forage-Job%20Simulation-075AAA?style=for-the-badge&labelColor=01295C" alt="Forage Job Simulation"/>
<img src="https://img.shields.io/badge/Python-3.13-ffffff?style=for-the-badge&logo=python&logoColor=075AAA&labelColor=01295C" alt="Python 3.13"/>
<img src="https://img.shields.io/badge/Model-Random%20Forest-EB2226?style=for-the-badge&labelColor=01295C" alt="Random Forest"/>
<img src="https://img.shields.io/badge/Validation-5--Fold%20Stratified%20CV-075AAA?style=for-the-badge&labelColor=01295C" alt="5-Fold Stratified CV"/>

<br/><br/>

<!-- ═══════════════════════ NAVIGATION ═══════════════════════ -->
<a href="#overview"><img src="https://img.shields.io/badge/Overview-01295C?style=for-the-badge" alt="Overview"/></a>
<a href="#problem"><img src="https://img.shields.io/badge/Business%20Problem-075AAA?style=for-the-badge" alt="Business problem"/></a>
<a href="#data"><img src="https://img.shields.io/badge/Data%20%26%20Prep-EB2226?style=for-the-badge" alt="Data and preparation"/></a>
<a href="#model"><img src="https://img.shields.io/badge/Model-01295C?style=for-the-badge" alt="Model"/></a>
<a href="#insights"><img src="https://img.shields.io/badge/Insights-075AAA?style=for-the-badge" alt="Insights"/></a>
<a href="#run"><img src="https://img.shields.io/badge/How%20To%20Run-EB2226?style=for-the-badge" alt="How to run"/></a>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ OVERVIEW ═══════════════════════ -->
<a id="overview"></a>

## 📌 Project Overview

This repository contains my work from the **British Airways Data Science Job Simulation**, completed through the **Forage** platform.

The project uses customer booking data to understand booking behaviour and to build a machine learning model that predicts **whether a customer is likely to complete a flight booking**.

The simulation gave me practical experience in **data exploration, feature engineering, machine learning, model validation, feature importance analysis, and communicating data-driven insights in a business context**.

```mermaid
flowchart LR
    A["🔎 Explore<br/>the data"] --> B["🧹 Prepare<br/>for ML"]
    B --> C["⚙️ Engineer<br/>features"]
    C --> D["🌲 Train<br/>Random Forest"]
    D --> E["✅ Validate<br/>5-Fold Stratified CV"]
    E --> F["📈 Feature<br/>importance"]
    F --> G["💼 Business<br/>insights"]
    style A fill:#01295c,stroke:none,color:#fff
    style B fill:#075aaa,stroke:none,color:#fff
    style C fill:#ffffff,stroke:#075aaa,color:#01295c
    style D fill:#eb2226,stroke:none,color:#fff
    style E fill:#075aaa,stroke:none,color:#fff
    style F fill:#ffffff,stroke:#075aaa,color:#01295c
    style G fill:#01295c,stroke:none,color:#fff
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ BUSINESS PROBLEM ═══════════════════════ -->
<a id="problem"></a>

## 🎯 Business Problem

Airlines need to identify potential customers **before they arrive at the airport**, rather than relying on last-minute purchasing decisions.

The objective of this project was to use historical customer booking data to:

- 🔍 Explore and understand customer booking behaviour
- 🧹 Prepare the dataset for machine learning
- ⚙️ Engineer relevant features that could improve model performance
- 🤖 Build a predictive classification model
- ✅ Evaluate the model using cross-validation and appropriate metrics
- 📈 Identify the variables that contributed most to the model's predictions
- 🎤 Communicate the findings through a business-focused presentation

> 🎯 **Target variable:** whether a customer completed a booking.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ DATA ═══════════════════════ -->
<a id="data"></a>

## 📊 Dataset

The project uses a **customer booking dataset** with information about customers' flight searches and booking behaviour.

<div align="center">

| ⏱️ **Timing** | 🗺️ **Route & Origin** | 🧳 **Trip** | 👥 **Booking** |
|:---:|:---:|:---:|:---:|
| Purchase lead time | Flight and route information | Length of stay | Number of passengers |
| | Booking origin | Trip characteristics | Booking-related behaviour |
| | | | **Booking outcome** |

</div>

The initial analysis looked at the dataset structure, data types, distributions and relevant statistics before the data was prepared for modelling.

## 🔎 Data Exploration & Preparation

<details open>
<summary><b>🧹 What the preparation process included</b></summary>
<br/>

- 🔬 Inspecting the dataset and understanding the available variables
- 🧾 Checking data types and dataset structure
- 📊 Examining the target variable and class distribution
- 🎛️ Identifying relevant numerical and categorical features
- 🛠️ Preparing variables for machine learning
- ✨ Creating additional features to represent useful customer and booking behaviour

</details>

### ⚙️ Feature Engineering

Additional features were created to give the model more informative representations of the data, while **keeping the original customer booking information**.

<div align="center">

| 🧩 Engineered Feature | 💬 What it represents |
|:---|:---|
| **`route_freq`** | The frequency of a particular route |
| **`booking_origin_freq`** | The frequency of bookings from a particular origin |
| **`lead_time_per_passenger`** | Purchase lead time combined with passenger count, showing planning behaviour relative to group size |

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ MODEL ═══════════════════════ -->
<a id="model"></a>

## 🤖 Machine Learning Model

### 🌲 Random Forest Classifier

A **Random Forest Classifier** was selected for the predictive modelling task. It is suitable for this kind of tabular classification problem because it can:

<div align="center">

| 🔀 Non-linear | 🤝 Interactions | 🧮 Mixed variables | 📏 Importance | 🌳 Stability |
|:---:|:---:|:---:|:---:|:---:|
| Capture non-linear relationships | Handle interactions between features | Work well with a mixture of predictive variables | Provide feature importance measurements | Reduce reliance on a single decision tree |

</div>

The model was trained to predict the customer's booking outcome.

### ⚖️ Class Imbalance

The target variable has an imbalance between the booking classes. To handle this during training, **balanced class weights** were used so the model gives greater consideration to the minority class.

```mermaid
flowchart LR
    A["⚖️ Imbalanced<br/>booking classes"] --> B["🎛️ Balanced<br/>class weights"] --> C["🎯 Minority class<br/>gets more consideration"]
    style A fill:#eb2226,stroke:none,color:#fff
    style B fill:#075aaa,stroke:none,color:#fff
    style C fill:#01295c,stroke:none,color:#fff
```

## 🧪 Model Validation

To check whether the model generalizes across different subsets of the data, **5-Fold Stratified Cross-Validation** was used. Stratification keeps a similar distribution of the target classes in each fold.

Validation was used to assess how **consistent** the model's performance is, rather than relying on a single train-test split.

```mermaid
flowchart LR
    D["📁 Dataset"] --> F1["Fold 1"]
    D --> F2["Fold 2"]
    D --> F3["Fold 3"]
    D --> F4["Fold 4"]
    D --> F5["Fold 5"]
    F1 & F2 & F3 & F4 & F5 --> R["📊 Cross-validation<br/>performance"]
    style D fill:#01295c,stroke:none,color:#fff
    style F1 fill:#075aaa,stroke:none,color:#fff
    style F2 fill:#075aaa,stroke:none,color:#fff
    style F3 fill:#075aaa,stroke:none,color:#fff
    style F4 fill:#075aaa,stroke:none,color:#fff
    style F5 fill:#075aaa,stroke:none,color:#fff
    style R fill:#eb2226,stroke:none,color:#fff
```

### 📏 Evaluation metrics

These metrics give a broader view of the model's predictive performance, especially when the target classes are imbalanced.

<div align="center">

<img src="https://img.shields.io/badge/Accuracy-01295C?style=for-the-badge" alt="Accuracy"/>
<img src="https://img.shields.io/badge/Precision-075AAA?style=for-the-badge" alt="Precision"/>
<img src="https://img.shields.io/badge/Recall-EB2226?style=for-the-badge" alt="Recall"/>
<img src="https://img.shields.io/badge/F1--score-01295C?style=for-the-badge" alt="F1-score"/>
<img src="https://img.shields.io/badge/Cross--Validation-075AAA?style=for-the-badge" alt="Cross-validation"/>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ INSIGHTS ═══════════════════════ -->
<a id="insights"></a>

## 📈 Feature Importance & Business Insights

Feature importance was analysed to understand which variables contributed most strongly to the Random Forest model's predictions.

```mermaid
flowchart LR
    A["⏱️ purchase_lead"] --> T(("🎯<br/>Booking<br/>outcome"))
    B["🗺️ route_freq"] --> T
    C["🌍 booking_origin_freq"] --> T
    D["🧳 length_of_stay"] --> T
    E["👥 lead_time_per_passenger"] --> T
    style A fill:#eb2226,stroke:none,color:#fff
    style B fill:#075aaa,stroke:none,color:#fff
    style C fill:#01295c,stroke:none,color:#fff
    style D fill:#075aaa,stroke:none,color:#fff
    style E fill:#01295c,stroke:none,color:#fff
    style T fill:#ffffff,stroke:#eb2226,stroke-width:3px,color:#01295c
```

<table>
<tr>
<td width="50%" valign="top">

### 1️⃣ ⏱️ Purchase Lead Time
`purchase_lead`

One of the **strongest predictive variables** in the model. It shows the time between the customer's booking activity and the planned flight, which gives useful information about booking behaviour.

</td>
<td width="50%" valign="top">

### 2️⃣ 🗺️ Route Frequency
`route_freq`

Contributed to the model's predictive ability by representing how often a particular route appeared in the dataset.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 3️⃣ 🌍 Booking Origin Frequency
`booking_origin_freq`

Captured how often bookings came from a particular origin, adding information about customer booking patterns.

</td>
<td width="50%" valign="top">

### 4️⃣ 🧳 Length of Stay
`length_of_stay`

Helped the model tell apart different booking behaviours and travel patterns.

</td>
</tr>
<tr>
<td colspan="2" valign="top">

### 5️⃣ 👥 Lead Time per Passenger
`lead_time_per_passenger`

This engineered feature gave an extra view of planning behaviour relative to the number of passengers in the booking.

</td>
</tr>
</table>

> [!NOTE]
> Feature importance shows how useful a variable was to the trained model's predictions. It does **not** establish that the variable directly causes a customer to make a booking.

## 💼 Business Interpretation

The model shows how historical customer booking data can be used to find patterns linked to booking outcomes. The feature importance analysis helps a business see which customer and booking characteristics are most useful for predictive modelling.

These insights can support **more proactive customer engagement**, by helping identify customers who may be more likely to complete a booking.

> [!IMPORTANT]
> Model predictions should be evaluated alongside business considerations and additional customer information before being used in a real-world decision-making system.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ TOOLS ═══════════════════════ -->
## 🛠️ Technologies & Tools

<div align="center">

<table align="center"><tr>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" height="48" alt="Python"/><br/><sub><b>Python 3.13</b></sub></td>
<td align="center" width="100"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/pandas/ffffff"><img src="https://cdn.simpleicons.org/pandas/150458" width="48" height="48" alt="Pandas"/></picture><br/><sub><b>Pandas</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="48" height="48" alt="NumPy"/><br/><sub><b>NumPy</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="48" height="48" alt="Scikit-learn"/><br/><sub><b>Scikit-learn</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" width="48" height="48" alt="Matplotlib"/><br/><sub><b>Matplotlib</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/seaborn/seaborn-original.svg" width="48" height="48" alt="Seaborn"/><br/><sub><b>Seaborn</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="48" height="48" alt="Jupyter"/><br/><sub><b>Jupyter</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" width="48" height="48" alt="VS Code"/><br/><sub><b>VS Code</b></sub></td>
<td align="center" width="100"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/github/ffffff"><img src="https://cdn.simpleicons.org/github/181717" width="48" height="48" alt="Git and GitHub"/></picture><br/><sub><b>Git & GitHub</b></sub></td>
</tr></table>

</div>

| Tool | Used for |
|:---|:---|
| 🐍 **Pandas** | Data manipulation and analysis |
| 🔢 **NumPy** | Numerical computing |
| 🤖 **Scikit-learn** | Machine learning and model evaluation |
| 📊 **Matplotlib** | Data visualization |
| 🎨 **Seaborn** | Statistical visualization |

## 📂 Project Structure

```text
British-Airways---Data-Science-Job-Simulation/
│
├── task1.py
├── task2.py
├── requirements.txt
├── README.md
│
└── ...
```

> The project structure may vary depending on the files included in the repository.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ HOW TO RUN ═══════════════════════ -->
<a id="run"></a>

## 🚀 How to Run the Project

**1️⃣ Clone the repository**

```bash
git clone https://github.com/paraskosambe-web/British-Airways---Data-Science-Job-Simulation.git
```

**2️⃣ Navigate to the project directory**

```bash
cd British-Airways---Data-Science-Job-Simulation
```

**3️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

**4️⃣ Run the analysis**

Open the project files in **Visual Studio Code** or **Jupyter Notebook** and execute the analysis. The project performs the required data preparation, feature engineering, model training, validation and visualization steps.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ DELIVERABLES ═══════════════════════ -->
## 📋 Project Deliverables

The simulation resulted in these key deliverables:

- [x] Exploratory analysis of the customer booking dataset
- [x] Prepared dataset for machine learning
- [x] Engineered predictive features
- [x] Random Forest classification model
- [x] Cross-validation and model evaluation
- [x] Feature importance analysis
- [x] Visualizations of model results
- [x] Business-focused summary of findings

## 🎓 Skills Demonstrated

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=17&duration=2200&pause=700&color=075AAA&center=true&vCenter=true&width=700&height=40&lines=Data+Exploration;Data+Cleaning+%26+Preparation;Feature+Engineering;Classification+with+Random+Forest;Cross-Validation+%26+Model+Evaluation;Feature+Importance+Analysis;Data+Visualization;Business+Insight+Generation;Communicating+Technical+Findings" alt="Skills animation"/>

<br/>

![Data Exploration](https://img.shields.io/badge/Data%20Exploration-01295C?style=for-the-badge)
![Data Preparation](https://img.shields.io/badge/Data%20Cleaning%20%26%20Preparation-075AAA?style=for-the-badge)
![Feature Engineering](https://img.shields.io/badge/Feature%20Engineering-EB2226?style=for-the-badge)
![Classification](https://img.shields.io/badge/Classification-01295C?style=for-the-badge)
![Random Forest](https://img.shields.io/badge/Random%20Forest-075AAA?style=for-the-badge)
![Cross-Validation](https://img.shields.io/badge/Cross--Validation-EB2226?style=for-the-badge)
![Model Evaluation](https://img.shields.io/badge/Model%20Evaluation-01295C?style=for-the-badge)
![Feature Importance](https://img.shields.io/badge/Feature%20Importance%20Analysis-075AAA?style=for-the-badge)
![Visualization](https://img.shields.io/badge/Data%20Visualization-EB2226?style=for-the-badge)
![Business Insight](https://img.shields.io/badge/Business%20Insight%20Generation-01295C?style=for-the-badge)
![Communication](https://img.shields.io/badge/Communicating%20Technical%20Findings-075AAA?style=for-the-badge)

</div>

## 🏢 About the Simulation

This project was completed as part of the **British Airways Data Science Job Simulation on Forage**. The simulation was a chance to work through a realistic data science workflow, from preparing customer booking data to building and evaluating a predictive machine learning model and communicating the results in a business context.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ REPOSITORY ═══════════════════════ -->
## 🔗 Repository

<div align="center">

<a href="https://github.com/paraskosambe-web/British-Airways---Data-Science-Job-Simulation"><img src="https://img.shields.io/badge/GitHub-View%20Repository-EB2226?style=for-the-badge&logo=github&logoColor=white&labelColor=01295C" alt="GitHub repository"/></a>
<img src="https://img.shields.io/badge/Platform-Forage%20%E2%80%A2%20British%20Airways%20Data%20Science-075AAA?style=for-the-badge&labelColor=01295C" alt="Forage platform"/>

<br/><br/>

**Paras Kosambe** • B.Sc. Computer Science Student • Aspiring Data Scientist

<a href="https://github.com/paraskosambe-web"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub"/></a>
<a href="https://www.linkedin.com/in/paras-kosambe"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="mailto:paraskosambe@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email"/></a>

<br/><br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=17&duration=3500&pause=1200&color=EB2226&center=true&vCenter=true&width=640&lines=Data+in.+Insight+out.+%E2%9C%88%EF%B8%8F;Predicting+bookings%2C+one+session+at+a+time." alt="Footer typing"/>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:01295c,55:075aaa,100:eb2226&height=120&section=footer" width="100%" alt="footer wave"/>
