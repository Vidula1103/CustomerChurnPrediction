export interface Customer {
  id: string;
  name: string;
  tenure: number;         // in months, e.g., 1 to 72
  monthlyCharges: number; // in USD, e.g., 20 to 150
  usageFactor: number;    // usage pattern, 0.1 to 1.5, representing ratio of usage compared to avg
  supportCalls: number;   // number of customer service calls, e.g., 0 to 10
  contractType: 'month-to-month' | 'one-year' | 'two-year';
  billingIssues: number;  // 1 if yes, 0 if no
  churned: number;        // 1 if left, 0 if loyal
  // Normalized feature array for ML models:
  // [tenure_norm, monthlyCharges_norm, usageFactor_norm, supportCalls_norm, contractType_MonthToMonth, contractType_OneYear, billingIssues]
  features?: number[];
}

export type ModelType = 'logistic_regression' | 'decision_tree' | 'neural_network';

export interface ModelParams {
  learningRate: number;
  epochs: number;
  regularization: number; // L2 coefficient for logistic regression
  threshold: number;      // Churn classifier probability decision threshold
  maxDepth: number;       // For Decision Tree
  hiddenUnits: number;    // Hidden layer units for neural network
}

export interface MetricPoint {
  threshold: number;
  tpr: number;
  fpr: number;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: MetricPoint[];
  featureImportance: { name: string; value: number }[];
}

export interface TrainingLog {
  epoch: number;
  loss: number;
  accuracy: number;
}
