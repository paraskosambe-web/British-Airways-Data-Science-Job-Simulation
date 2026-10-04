<!-- ═══════════════════════ HEADER ═══════════════════════ -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:01295c,55:075aaa,100:eb2226&height=290&section=header&text=Customer%20Booking%20Prediction&fontSize=48&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=British%20Airways%20%E2%80%A2%20Data%20Science%20Job%20Simulation%20%E2%80%A2%20Task%202&descSize=19&descAlignY=58" width="100%" alt="British Airways Customer Booking Prediction"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=21&duration=3000&pause=900&color=EB2226&center=true&vCenter=true&width=800&height=50&lines=%E2%9C%88%EF%B8%8F+Will+this+customer+complete+the+booking%3F;%F0%9F%93%8A+Learning+from+50%2C000+customer+booking+sessions;%F0%9F%8C%B2+Random+Forest+%2B+5-Fold+Stratified+Cross-Validation;%F0%9F%92%A1+Turning+booking+data+into+business+insight" alt="Typing animation"/>

<br/>

<img src="https://img.shields.io/badge/Forage-Job%20Simulation-075AAA?style=for-the-badge&labelColor=01295C" alt="Forage Job Simulation"/>
<img src="https://img.shields.io/badge/Task-2%20Predictive%20Modeling-EB2226?style=for-the-badge&labelColor=01295C" alt="Task 2"/>
<img src="https://img.shields.io/badge/Python-3.13-ffffff?style=for-the-badge&logo=python&logoColor=075AAA&labelColor=01295C" alt="Python 3.13"/>
<img src="https://img.shields.io/badge/Model-Random%20Forest-075AAA?style=for-the-badge&labelColor=01295C" alt="Random Forest"/>

<br/><br/>

<!-- ═══════════════════════ NAVIGATION ═══════════════════════ -->
<a href="#glance"><img src="https://img.shields.io/badge/At%20a%20Glance-01295C?style=for-the-badge" alt="At a glance"/></a>
<a href="#context"><img src="https://img.shields.io/badge/Business%20Context-075AAA?style=for-the-badge" alt="Business context"/></a>
<a href="#method"><img src="https://img.shields.io/badge/Methodology-EB2226?style=for-the-badge" alt="Methodology"/></a>
<a href="#insights"><img src="https://img.shields.io/badge/Key%20Insights-01295C?style=for-the-badge" alt="Key insights"/></a>
<a href="#run"><img src="https://img.shields.io/badge/How%20To%20Run-075AAA?style=for-the-badge" alt="How to run"/></a>
<a href="#deliverables"><img src="https://img.shields.io/badge/Deliverables-EB2226?style=for-the-badge" alt="Deliverables"/></a>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ AT A GLANCE ═══════════════════════ -->
<a id="glance"></a>

## ✈️ At a Glance

This repository contains the machine learning solution for **Task 2** of the **British Airways Data Science Job Simulation** on Forage. The goal is to build a predictive model that uncovers customer booking patterns and identifies the key factors that drive flight booking completions.

> 💡 **In one line:** *Learn from how customers behave while browsing, and predict whether each session ends in a completed booking.*

<div align="center">

| 📁 **Dataset** | 🎯 **Booking Rate** | 🌲 **Model** | ✅ **Validation** |
|:---:|:---:|:---:|:---:|
| **50,000**<br/>customer booking sessions | **~15%**<br/>complete a booking | **Random Forest**<br/>Classifier | **5-Fold**<br/>Stratified Cross-Validation |

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ BUSINESS CONTEXT ═══════════════════════ -->
<a id="context"></a>

## 📌 Business Context & Objective

British Airways wants to **increase conversion rates** across its direct booking channels. When customers browse flight schedules, choose travel add-ons or select routes, their digital behavior gives strong signals about whether they will complete a reservation.

<table>
<tr>
<td width="33%" valign="top">

### 🎯 1. Predictive Analytics
Build a **binary classification model** that predicts whether a customer session ends in a completed booking (`booking_complete`).

</td>
<td width="33%" valign="top">

### 🔍 2. Commercial Insights
Uncover the main **behavioral drivers and preferences**, such as lead times, extra services and trip duration, that influence purchase decisions.

</td>
<td width="34%" valign="top">

### 🧭 3. Strategic Recommendations
Turn model insights into **actionable strategies** for marketing, pricing and product design teams to lift overall booking conversions.

</td>
</tr>
</table>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ METHODOLOGY ═══════════════════════ -->
<a id="method"></a>

## 🛠️ Methodological Approach

```mermaid
flowchart LR
    A["📁 Raw data<br/>50,000 sessions"] --> B["🧹 Cleaning<br/>& EDA"]
    B --> C["⚙️ Feature<br/>engineering"]
    C --> D["🌲 Random Forest<br/>training"]
    D --> E["✅ 5-Fold Stratified<br/>validation"]
    E --> F["💡 Insights &<br/>recommendations"]
    style A fill:#01295c,stroke:none,color:#fff
    style B fill:#075aaa,stroke:none,color:#fff
    style C fill:#ffffff,stroke:#075aaa,color:#01295c
    style D fill:#eb2226,stroke:none,color:#fff
    style E fill:#075aaa,stroke:none,color:#fff
    style F fill:#01295c,stroke:none,color:#fff
```

<details open>
<summary><b>🧹 1. Data Cleaning & Exploratory Data Analysis (EDA)</b></summary>
<br/>

- **Data integrity and consistency:** inspected the quality of all 50,000 records, checked for missing values, made variable encoding consistent, and converted text categories into numbers.
- **Class imbalance:** analyzed the target variable and found a natural imbalance of about **15% completed bookings** versus **85% non-completions**.

```mermaid
%%{init: {'theme':'base','themeVariables':{'pie1':'#eb2226','pie2':'#075aaa','pieTitleTextSize':'18px','pieSectionTextColor':'#ffffff'}}}%%
pie showData title Target variable: booking_complete (approx.)
    "Completed booking" : 15
    "No booking" : 85
```

</details>

<details open>
<summary><b>⚙️ 2. Feature Engineering</b></summary>
<br/>

Domain-specific features were created to capture more detailed customer behavior:

| 🧩 Feature group | What it captures |
|:---|:---|
| ⏱️ **Planning & timing indicators** | Customer lead-time ratios and group-booking dynamics |
| 📈 **Aggregated demand metrics** | Historical route frequencies and origin-market demand, to give context to high-volume customer segments |
| 🧳 **Service bundling** | Combinations of add-ons (extra baggage, preferred seat, in-flight meals) to gauge purchase intent |

</details>

<details open>
<summary><b>🌲 3. Model Building & Validation</b></summary>
<br/>

- **Algorithm:** a **Random Forest Classifier**, chosen for its strength with non-linear customer behavior, feature interactions and tabular data.
- **Validation:** **5-Fold Stratified Cross-Validation**, so the model generalizes well across imbalanced splits without overfitting.
- **Balanced class weights:** decision-tree weighting was adjusted during training to handle the imbalance without losing reliability on successful bookings.

</details>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ INSIGHTS ═══════════════════════ -->
<a id="insights"></a>

## 💡 Key Business Drivers & Strategic Insights

Feature importance analysis highlighted **four main levers** that decide whether a customer completes a booking.

```mermaid
flowchart LR
    L["⏱️ purchase_lead"] --> T(("🎯<br/>booking_complete"))
    R["🗺️ route_freq<br/>booking_origin_freq"] --> T
    S["🧳 length_of_stay"] --> T
    G["👨‍👩‍👧 lead_time_per_passenger"] --> T
    style L fill:#eb2226,stroke:none,color:#fff
    style R fill:#075aaa,stroke:none,color:#fff
    style S fill:#01295c,stroke:none,color:#fff
    style G fill:#075aaa,stroke:none,color:#fff
    style T fill:#ffffff,stroke:#eb2226,stroke-width:3px,color:#01295c
```

<table>
<tr>
<td width="50%" valign="top">

### 1️⃣ ⏱️ Purchase Lead Time
`purchase_lead`

How far in advance a customer books is the **single strongest indicator** of purchase intent. Customers who book far ahead behave very differently from last-minute browsers.

</td>
<td width="50%" valign="top">

### 2️⃣ 🗺️ Route & Origin Demand
`route_freq` • `booking_origin_freq`

Market location and route volume strongly affect how quickly customers convert. High-density origin markets reflect strong organic demand.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 3️⃣ 🧳 Trip Duration
`length_of_stay`

Length of stay correlates strongly with conversion. Long-haul and holiday trips involve different decision windows than short trips.

</td>
<td width="50%" valign="top">

### 4️⃣ 👨‍👩‍👧 Group Lead Dynamics
`lead_time_per_passenger`

Planning timelines change with party size, separating solo business travelers from family holiday bookings.

</td>
</tr>
</table>

<div align="center">

**📊 Feature importance**

<img src="./visuals/feature_importance.png" width="85%" alt="Feature importance chart"/>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ STRUCTURE ═══════════════════════ -->
## 📂 Repository Structure

```text
british-airways-data-science/
├── data/
│   └── customer_booking.csv                   # Raw dataset: 50,000 customer booking sessions
├── notebooks/
│   └── customer_booking_prediction.ipynb      # Cleaning, EDA and modeling, end to end
├── visuals/
│   └── feature_importance.png                 # Relative feature importance drivers
├── deliverables/
│   └── British_Airways_Task2_Executive_Summary.pptx   # Non-technical stakeholder presentation
├── requirements.txt                           # Python dependencies
└── README.md                                  # Overview and documentation
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ TECH STACK ═══════════════════════ -->
## 🧰 Tech Stack & Environment

<div align="center">

<table align="center"><tr>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48" height="48" alt="Python"/><br/><sub><b>Python 3.13</b></sub></td>
<td align="center" width="100"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/pandas/ffffff"><img src="https://cdn.simpleicons.org/pandas/150458" width="48" height="48" alt="Pandas"/></picture><br/><sub><b>Pandas</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="48" height="48" alt="NumPy"/><br/><sub><b>NumPy</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="48" height="48" alt="Scikit-Learn"/><br/><sub><b>Scikit-Learn</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" width="48" height="48" alt="Matplotlib"/><br/><sub><b>Matplotlib</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/seaborn/seaborn-original.svg" width="48" height="48" alt="Seaborn"/><br/><sub><b>Seaborn</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="48" height="48" alt="Jupyter"/><br/><sub><b>Jupyter</b></sub></td>
<td align="center" width="100"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vscode/vscode-original.svg" width="48" height="48" alt="VS Code"/><br/><sub><b>VS Code</b></sub></td>
</tr></table>

</div>

| Purpose | Tools |
|:---|:---|
| 🐍 **Language** | Python 3.13 |
| 🧮 **Data analysis & processing** | `pandas`, `numpy` |
| 🤖 **Machine learning & evaluation** | `scikit-learn` |
| 📊 **Visualization** | `matplotlib`, `seaborn` |
| 💻 **Environment** | Visual Studio Code / Jupyter Notebooks |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ HOW TO RUN ═══════════════════════ -->
<a id="run"></a>

## 🚀 How to Run the Project

**1️⃣ Clone the repository**

```bash
git clone https://github.com/paraskosambe-web/British-Airways---Data-Science-Job-Simulation.git
cd British-Airways---Data-Science-Job-Simulation
```

**2️⃣ Install the dependencies**

```bash
pip install -r requirements.txt
```

**3️⃣ Run the analysis**

Launch **VS Code** or **Jupyter Lab**, open the notebook below, and **run all cells**. This performs the data processing, trains the Random Forest model and generates the visual outputs.

```text
notebooks/customer_booking_prediction.ipynb
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ DELIVERABLES ═══════════════════════ -->
<a id="deliverables"></a>

## 📋 Project Deliverables

<div align="center">

| 📦 Deliverable | What it is |
|:---|:---|
| 📓 **[Jupyter Notebook](./notebooks/customer_booking_prediction.ipynb)** | End-to-end data cleaning, EDA, feature engineering and modeling |
| 📊 **[Feature Importance Chart](./visuals/feature_importance.png)** | Visual of the key drivers behind booking completions |
| 🎤 **[Executive Summary (PPTX)](./deliverables/British_Airways_Task2_Executive_Summary.pptx)** | Non-technical presentation for stakeholders |

<br/>

<a href="./notebooks/customer_booking_prediction.ipynb"><img src="https://img.shields.io/badge/Open%20the%20Notebook-%E2%86%92-EB2226?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=01295C" alt="Open notebook"/></a>
<a href="./deliverables/British_Airways_Task2_Executive_Summary.pptx"><img src="https://img.shields.io/badge/Executive%20Summary-%E2%86%92-075AAA?style=for-the-badge&labelColor=01295C" alt="Executive summary"/></a>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:eb2226,50:ffffff,100:075aaa&height=3" width="100%" alt="divider"/>

<!-- ═══════════════════════ AUTHOR ═══════════════════════ -->
## 👨‍💻 Author

<div align="center">

### Paras Kosambe
**B.Sc. Computer Science Student** • Aspiring Data Scientist • AI/ML • Python • SQL

<br/>

<a href="https://github.com/paraskosambe-web"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
<a href="https://www.linkedin.com/in/paras-kosambe"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="mailto:paraskosambe@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/></a>

<br/><br/>

<sub>This project was completed as part of a Forage job simulation for educational purposes. It is not affiliated with or endorsed by British Airways.</sub>

<br/><br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=17&duration=3500&pause=1200&color=075AAA&center=true&vCenter=true&width=640&lines=Data+in.+Insight+out.+%E2%9C%88%EF%B8%8F;Predicting+bookings%2C+one+session+at+a+time." alt="Footer typing"/>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:01295c,55:075aaa,100:eb2226&height=120&section=footer" width="100%" alt="footer wave"/>
