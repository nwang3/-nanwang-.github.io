import pandas as pd
from openpyxl import Workbook
import numpy as np
import matplotlib.pyplot as plt

def add_sample_data(ws, property_name, street_name):
    ws.append(['Year', 'Gross Rental Income', 'Operating Expenses', 'NOI', 'Cap Rate', 'Cash-on-Cash Return', 'DSCR', 'IRR', 'Vacancy Rate', 'Property Name', 'Street Name'])
    for year in range(1, 11):
        gross_rental_income = np.random.randint(800000, 1200000)
        operating_expenses = np.random.randint(300000, 500000)
        noi = gross_rental_income - operating_expenses
        current_market_value = np.random.randint(5000000, 8000000)
        cap_rate = noi / current_market_value
        equity_investment = np.random.randint(1500000, 2500000)
        pre_tax_cash_flow = np.random.randint(200000, 400000)
        cash_on_cash_return = pre_tax_cash_flow / equity_investment
        debt_service = np.random.randint(100000, 200000)
        dscr = noi / debt_service
        irr = np.random.uniform(0.08, 0.15)
        vacancy_rate = np.random.uniform(0.02, 0.1)

        ws.append([year, gross_rental_income, operating_expenses, noi, cap_rate, cash_on_cash_return, dscr, irr, vacancy_rate, property_name, street_name])

# Define street names for each property
street_names = {'Property1': 'Main Street', 'Property2': 'Oak Avenue', 'Property3': 'Maple Lane'}

for property_name, street_name in street_names.items():
    wb = Workbook()
    ws = wb.active
    ws.title = street_name
    add_sample_data(ws, property_name, street_name)
    wb.save(f'{street_name}.xlsx')

dfs = []
for street_name in street_names.values():
    df = pd.read_excel(f'{street_name}.xlsx')
    dfs.append(df)

combined = pd.concat(dfs)

# Rearrange columns with "Property Name" as the leftmost column
combined = combined[['Property Name', 'Year', 'Gross Rental Income', 'Operating Expenses', 'NOI', 'Cap Rate', 'Cash-on-Cash Return', 'DSCR', 'IRR', 'Vacancy Rate', 'Street Name']]

# Add analysis columns
combined['YOY% Gross Rental Income'] = combined.groupby('Property Name')['Gross Rental Income'].pct_change() * 100
combined['YOY% NOI'] = combined.groupby('Property Name')['NOI'].pct_change() * 100
combined['YOY% Cap Rate'] = combined.groupby('Property Name')['Cap Rate'].pct_change() * 100
combined['YOY% Cash-on-Cash Return'] = combined.groupby('Property Name')['Cash-on-Cash Return'].pct_change() * 100

combined.to_excel('Consolidated Asset Management Sheet.xlsx', index=False)

if 'IRR' in combined.columns and 'Cash-on-Cash Return' in combined.columns:
    irr = combined['IRR'].values
    cash_on_cash_return = combined['Cash-on-Cash Return'].values

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(irr, bins=20, edgecolor='black')
    ax1.set_title('IRR Distribution')
    ax1.set_xlabel('IRR')
    ax1.set_ylabel('Frequency')

    ax2.hist(cash_on_cash_return, bins=20, edgecolor='black')
    ax2.set_title('Cash-on-Cash Return Distribution')
    ax2.set_xlabel('Cash-on-Cash Return')
    ax2.set_ylabel('Frequency')

    fig.suptitle('Key Performance Metrics')
    plt.savefig('analysis.png') 
    plt.show() 

    print('Analysis complete')
else:
    print("Columns 'IRR' and 'Cash-on-Cash Return' not found in the combined DataFrame.")
