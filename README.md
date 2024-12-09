# Clustering Models Project

## 📜 Project Overview

This project aims to address one of the major challenges in a global trading company: better understanding its customer base, products, and operations. Through data analysis and the development of clustering models, we seek to identify patterns, trends, and key factors influencing the company’s success.

We will work with a dataset that includes information on sales, shipping, costs, and profits at the customer and product levels. This will allow us to design specific strategies that maximize profits and optimize operational processes.

## 🔧️ Project Structure
```
Global-Trade-Insights-Clustering-Project
├── data/                               # Folder containing datasets
├── notebook/                           # Jupyter Notebooks for EDA and clustering
├── src/                                # Source code
├── .gitignore                          # Git ignore file
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies
```

## 🔧 Installation and Requirements

This project was developed in Python 3.12. To set it up, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SupernovaIa/Proyecto-9-Global-Trade-Insights-Clustering-and-Regression
   ```
2. Navigate to the project directory:
   ```bash
   cd Proyecto-9-Global-Trade-Insights-Clustering-and-Regression
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebooks:
   Perform exploratory analysis and clustering by executing the notebooks in the `notebook/` folder.

### Required Libraries:

- **Pandas:** Data manipulation and analysis ([Documentation](https://pandas.pydata.org/)) - `pandas==2.2.2`.
- **NumPy:** Numerical data processing ([Documentation](https://numpy.org/)) - `numpy==1.26.4`.
- **Matplotlib:** Data visualization ([Documentation](https://matplotlib.org/)) - `matplotlib==3.9.2`.
- **Seaborn:** Statistical visualizations ([Documentation](https://seaborn.pydata.org/)) - `seaborn==0.13.2`.
- **Category Encoders:** Encoding categorical variables ([Documentation](https://contrib.scikit-learn.org/category_encoders/)) - `category-encoders==2.6.0`.
- **Scikit-learn:** Machine Learning algorithms ([Documentation](https://scikit-learn.org/)) - `scikit-learn==1.3.2`.
- **SciPy:** Scientific computations ([Documentation](https://scipy.org/)) - `scipy==1.14.1`.
- **Yellowbrick:** Visualizations for Machine Learning models ([Documentation](https://www.scikit-yb.org/en/latest/)) - `yellowbrick==1.5`.
- **Plotly:** Interactive visualizations ([Documentation](https://plotly.com/)) - `plotly==5.24.1`.
- **Statsmodels:** Statistical modeling ([Documentation](https://www.statsmodels.org/)) - `statsmodels==0.14.4`.

## 📊 Results and Conclusions

### Exploratory Data Analysis

#### General Description
The dataset consists of individual transactions that include information about orders, customers, and products. To facilitate analysis, the data needs to be grouped into specific tables for customers, products, and orders.

#### Numerical Variables
- **`Sales`**: Wide range (1 to 20,000, average 246.49). Outliers in high sales (>5,000).
- **`Quantity`**: Ranges from 1 to 14 units, with an average of 3–4. Outliers for quantities >10.
- **`Discount`**: Average 14% (maximum 85%). Outliers for discounts >50%.
- **`Profit`**: Wide variability (-6,600 to 8,400). Outliers at both extremes.
- **`ShippingCost`**: Mostly low costs. Outliers for values >200.

#### Categorical Variables
- **`OrderID`**: 25,035 unique orders, some involving multiple transactions.
- **`ShipMode`**: `Standard Class` is the most common (60%), while `Same Day` is the least frequent (5%).
- **`Segment`**: Three categories: `Consumer` (52%), `Corporate`, and `Home Office`.
- **`Market` and `Region`**: APAC and Canada show positive metrics.
- **`Category` and `Sub-Category`**: Technology has the best performance, while Office Supplies generates lower profits.


#### Groupings
- **Customers**: 795 unique customers. Each customer is associated with a single segment and exhibits high geographic diversity. Relevant columns: 
  - `Segment`
  - Number of unique products (`ProductID`)
  - Total `Sales`, `Quantity`, `Profit`, and `ShippingCost`.

- **Products**: 10,292 unique products. Each product has a unique category and sub-category. Relevant columns:
  - `Category`
  - `Sub-Category`
  - Number of unique customers (`CustomerID`)
  - Total `Sales`, `Quantity`, `Profit`, and `ShippingCost`.

- **Orders**: 25,035 unique orders. Most orders are linked to a single customer and shipping mode. Relevant columns:
  - `ShipMode`
  - `OrderPriority`
  - Number of unique products (`ProductID`)
  - Total `Sales`, `Quantity`, `Profit`, and `ShippingCost`.

#### Correlations
- **Sales and Shipping Cost**: Significant positive correlation.
- **Sales and Quantity**: Moderate positive correlation.
- **Profit and Sales**: Moderate positive correlation.
- **Profit and Discount**: Moderate negative correlation, highlighting that higher discounts directly reduce profits.

### Clustering Analysis

To enhance insights, we applied clustering techniques across the three groupings (customers, products, and orders). Several algorithms were tested, including k-means, hierarchical clustering, spectral clustering, and DBSCAN. Data preprocessing varied depending on the grouping, and the best-performing configurations are summarized below.

#### Customers
- **Preprocessing**: Frequency encoding for the categorical variable `Segment` and MinMax scaling.
- **Results**: All clustering methods yielded the same result—two primary clusters differentiated by `Segment`. No significant differences were observed in other variables, limiting actionable insights.

#### Products
- **Preprocessing**: Removed `Sub-Category` (subset of `Category`), applied frequency encoding to `Category`, and used MinMax scaling.
- **Results**: Only k-means provided meaningful clusters, identifying two groups:
  - **Cluster 1**: Products in `Technology` and `Furniture`.
  - **Cluster 2**: Products in `Office Supplies`.
- **Insights**: 
  - Cluster 2 includes more products and customers but generates lower sales and profits.
  - This aligns with the characteristics of Office Supplies—lower-cost items purchased in bulk compared to higher-value items in Technology and Furniture.

#### Orders
- **Preprocessing**: Frequency encoding for `ShipMode` and `OrderPriority`, with MinMax scaling.
- **Results**: k-means was the only method producing meaningful clusters, resulting in two groups:
  - **Cluster 1**: Orders with significantly higher sales but associated losses.
  - **Cluster 2**: Orders with moderate sales and positive profits.
- **Insights**: 
  - The first cluster represents underperforming orders that merit closer analysis to improve business performance.

### Recommendations

#### 1. Optimize Discounts
Discounts have a noticeable negative impact on profits, so it is essential to revise current practices. By focusing on offering discounts selectively, such as targeting strategic customers or high-margin products, the business can reduce the financial impact while maintaining competitiveness.


#### 2. Focus on Profitable Products
Technology and Furniture stand out as the primary drivers of sales and profit, while Office Supplies, despite their high volume, contribute less to overall profitability. Marketing efforts should prioritize Technology and Furniture, with Office Supplies positioned as an entry point to attract and retain customers.

#### 3. Analyze Loss-Making Orders
A significant number of orders show high sales volumes but result in financial losses, likely due to high costs or aggressive discounting. Identifying and addressing the root causes of these losses will help enhance overall profitability and inform future operational strategies.

## 🧬 Next Steps

#### 1. Root cause analysis
Conduct deeper analyses to uncover the root causes and patterns behind the identified issues. This will provide actionable insights to address specific problem areas and enhance business performance.

#### 2. Sales and profit forecasting
Develop predictive models using time series analysis to forecast sales and profits. These models will help anticipate trends, optimize decision-making, and enable the business to adapt proactively to future market demands.

## 🤝 Contributions

Contributions are welcome. Follow these steps to collaborate:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request when ready.

If you have suggestions or improvements, feel free to open an issue.

## ✍️ Author

Javier Carreira - Lead Developer\
GitHub: [SupernovaIa](https://github.com/SupernovaIa)
