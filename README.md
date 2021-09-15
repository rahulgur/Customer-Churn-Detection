# Customer-Churn-Detection
problem statement:

A manager at the bank is disturbed with more and more customers leaving their credit card services.

They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

Data Dictionary:

Clientnum -> Client number. Unique identifier for the customer holding the account

Customer_Age -> Customer's Age in Years

Gender -> M=Male, F=Female

Dependent_count -> Number of dependents

Education_Level -> Educational Qualification of the account holder (example: high school, college graduate, etc.)

Marital_Status -> Married, Single, Unknown

Income_Category -> Annual Income Category of the account holder (<  40𝐾, 40K - 60K,  60𝐾− 80K,  80𝐾− 120K, > $120K, Unknown)

Card_Category -> Type of Card (Blue, Silver, Gold, Platinum)

Months_on_book -> Months on book (Time of Relationship)

Total_Relationship_Count -> Total no. of products held by the customer

Months_Inactive_12_mon -> No. of months inactive in the last 12 months

Contacts_Count_12_mon -> No. of Contacts in the last 12 months

Credit_Limit -> Credit Limit on the Credit Card

Total_Revolving_Bal -> Total Revolving Balance on the Credit Card

Avg_Open_To_Buy -> Open to Buy Credit Line (Average of last 12 months)

Total_Amt_Chng_Q4_Q1 -> Change in Transaction Amount (Q4 over Q1)

Total_Trans_Amt -> Total Transaction Amount (Last 12 months)

Total_Trans_Ct -> Total Transaction Count (Last 12 months)

Total_Ct_Chng_Q4_Q1 -> Change in Transaction Count (Q4 over Q1)

Avg_Utilization_Ratio -> Average Card Utilization Ratio

Insights:

1)The average age of all the customers is 46 years and among all the customers ,minimum age is 26 years and maximum age is 73 years.

2)Among all the customers,there is average of 2 Dependent_count and dependents range from 0 to 5.

3)customers have average of 35 months in relationship with that Creditcardservice and minimum relationship of 13 months and maximum relationship of 56 months.

4)There is average of 3 products held by the customer and minimum of 1 product and maximum of 6 products using that credit card.

5)customers are inactive with average of 2 months and there are also customers who are inactive for a peroid of 6 months.

6)The average credit limit for all the customers is 8631k dollars and max credit limit based on some customer Income is given as 34516k dollars.

7)The maximum change in transaction amount(q4 over q1) has 3.39 dollars.

8)The average total transaction amount done by customers from last 12 months is 4404k dollars and maximum amount is of 18484k dollars.

9)There is an average of 64 total transactions done by customers from the last 12 months and minimum of 10 transactions,maximum of 139 transactions from last 12 months.

10)The average card utilization ratio of all the customers is 0.27


Deployment:

The Model Pipeline is deployed on Aws Cloud by creating the endpoint and lambda is created to invoke the endpoint which has been created.
