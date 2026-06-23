import { Customer } from '../types';

const FIRST_NAMES = [
  'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Oliver', 'Sophia', 'Elijah', 'Isabella', 'James',
  'Mia', 'Benjamin', 'Charlotte', 'Lucas', 'Amelia', 'Alexander', 'Harper', 'Mason', 'Evelyn', 'Michael',
  'Daniel', 'Evelyn', 'Henry', 'Abigail', 'Alexander', 'Ella', 'Matthew', 'Scarlett', 'Jackson', 'Aria'
];

const LAST_NAMES = [
  'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
  'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
  'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson'
];

export function generateCustomers(size: number = 200, targetChurnRatio: number = 0.15): Customer[] {
  const customers: Customer[] = [];
  
  for (let i = 0; i < size; i++) {
    const id = `CUST-${1000 + i}`;
    const name = `${FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)]} ${LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)]}`;
    
    // Core attributes
    const contractRand = Math.random();
    const contractType: Customer['contractType'] = 
      contractRand < 0.65 ? 'month-to-month' : 
      contractRand < 0.85 ? 'one-year' : 'two-year';
      
    const supportCalls = Math.floor(Math.random() * 4) + (Math.random() < 0.15 ? Math.floor(Math.random() * 5) + 3 : 0);
    const tenure = contractType === 'month-to-month' 
      ? Math.floor(Math.random() * 12) + 1 
      : Math.floor(Math.random() * 60) + 12;
      
    const monthlyCharges = Math.floor(Math.random() * 80) + 30 + (supportCalls * 5);
    const usageFactor = Math.max(0.1, Math.min(1.5, Number((Math.random() * 0.8 + 0.5 - (supportCalls * 0.08)).toFixed(2))));
    const billingIssues = Math.random() < 0.25 ? 1 : 0;
    
    // Mathematically determine churn probability based on historical churn behaviors
    let score = -1.5; // Base log-odds (highly stable, low churn)
    
    // Risk factors
    if (contractType === 'month-to-month') score += 1.8;
    if (contractType === 'one-year') score += 0.5;
    score += (supportCalls * 0.7); // High support call is extreme churn trigger
    if (tenure < 6) score += 1.4;
    else if (tenure < 18) score += 0.6;
    if (usageFactor < 0.4) score += 1.2;
    if (monthlyCharges > 100) score += 0.6;
    if (billingIssues === 1) score += 0.9;
    
    // Sigmoid probability
    const prob = 1 / (1 + Math.exp(-score));
    
    // We adjust final binary label to approximate our target churn ratio
    // If contract is multi-year, they rarely churn unless score is incredibly high
    const churnThresholdAdjusted = 1 - targetChurnRatio * 1.5;
    const churned = prob > Math.random() && Math.random() < 0.85 ? 1 : 0;
    
    customers.push({
      id,
      name,
      tenure,
      monthlyCharges,
      usageFactor,
      supportCalls,
      contractType,
      billingIssues,
      churned
    });
  }
  
  // Enforce some class imbalance check & correction if necessary to resemble the setting
  const actualChurns = customers.filter(c => c.churned === 1).length;
  const currentRatio = actualChurns / size;
  
  // Sort and force adjust if we are significantly off the target ratio to make sure SMOTE actually has imbalanced data to preprocess
  if (Math.abs(currentRatio - targetChurnRatio) > 0.05) {
    // Sort by risk score (usage calls + contract + charges / tenure)
    const scoredCustomers = customers.map(c => {
      let r = c.supportCalls * 2 + (c.contractType === 'month-to-month' ? 3 : 0) + (c.tenure < 10 ? 2 : 0) + (c.billingIssues ? 1.5 : 0);
      return { c, r };
    }).sort((a, b) => b.r - a.r);
    
    const targetCount = Math.round(size * targetChurnRatio);
    for (let k = 0; k < size; k++) {
      scoredCustomers[k].c.churned = k < targetCount ? 1 : 0;
    }
  }
  
  return customers;
}

export const FEATURE_NAMES = [
  'Tenure (Months)',
  'Monthly Charges ($)',
  'Usage Pattern Factor',
  'Support Calls',
  'Contract: Month-to-Month',
  'Contract: One-Year',
  'Billing Issues'
];

export function extractFeatureVector(customer: Customer): number[] {
  return [
    customer.tenure,
    customer.monthlyCharges,
    customer.usageFactor,
    customer.supportCalls,
    customer.contractType === 'month-to-month' ? 1 : 0,
    customer.contractType === 'one-year' ? 1 : 0,
    customer.billingIssues
  ];
}

export function normalizeDataset(X: number[][]): {
  scaledX: number[][];
  mins: number[];
  maxs: number[];
} {
  if (X.length === 0) return { scaledX: [], mins: [], maxs: [] };
  const numFeatures = X[0].length;
  const mins = Array(numFeatures).fill(Infinity);
  const maxs = Array(numFeatures).fill(-Infinity);
  
  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < numFeatures; j++) {
      if (X[i][j] < mins[j]) mins[j] = X[i][j];
      if (X[i][j] > maxs[j]) maxs[j] = X[i][j];
    }
  }
  
  const scaledX = X.map(row => 
    row.map((val, idx) => {
      const denom = maxs[idx] - mins[idx];
      return denom === 0 ? 0 : (val - mins[idx]) / denom;
    })
  );
  
  return { scaledX, mins, maxs };
}

export function scaleSingleVector(vector: number[], mins: number[], maxs: number[]): number[] {
  return vector.map((val, idx) => {
    const denom = maxs[idx] - mins[idx];
    return denom === 0 ? 0 : (val - mins[idx]) / denom;
  });
}
