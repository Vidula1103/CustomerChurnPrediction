import React, { useMemo } from 'react';
import { ModelMetrics, TrainingLog, ModelType } from '../types';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line, Legend } from 'recharts';
import { Percent, TrendingDown, Target, Award, BarChart3, Activity, ListOrdered, CheckCircle2 } from 'lucide-react';

interface MetricsPanelProps {
  metrics: ModelMetrics | null;
  logs: TrainingLog[];
  modelType: ModelType;
  isSMOTEApplied: boolean;
}

export default function MetricsPanel({
  metrics,
  logs,
  modelType,
  isSMOTEApplied
}: MetricsPanelProps) {

  // Calculate estimated ROC AUC (Area Under Curve) using trapezoidal integration
  const aucScore = useMemo(() => {
    if (!metrics || !metrics.rocCurve || metrics.rocCurve.length === 0) return 0.5;
    
    // Ensure points are sorted by False Positive Rate
    const sorted = [...metrics.rocCurve].sort((a, b) => a.fpr - b.fpr);
    
    let area = 0;
    for (let i = 0; i < sorted.length - 1; i++) {
      const p1 = sorted[i];
      const p2 = sorted[i + 1];
      // Trapezoid area: height * (base1 + base2) / 2
      const baseDiff = p2.fpr - p1.fpr;
      const heightAvg = (p1.tpr + p2.tpr) / 2;
      area += baseDiff * heightAvg;
    }
    return Math.min(1, Math.max(0.5, area));
  }, [metrics]);

  if (!metrics) {
    return (
      <div className="bg-slate-50 border border-dashed border-slate-200 rounded-xl p-8 text-center" id="metrics-awaiting-training">
        <TrendingDown className="h-8 w-8 text-slate-300 mx-auto mb-2" />
        <h3 className="text-sm font-semibold text-slate-600">No Classifier Trained Yet</h3>
        <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
          Please select an algorithm configuration above, toggle SMOTE preprocessing, and click "Train Churn Classifier" to output model heuristics.
        </p>
      </div>
    );
  }

  const { accuracy, precision, recall, f1Score, confusionMatrix, featureImportance } = metrics;
  const { tp, tn, fp, fn } = confusionMatrix;
  const totalPredicted = tp + tn + fp + fn;

  const scoreCards = [
    { name: 'Model Accuracy', val: accuracy, color: 'text-blue-600 bg-blue-50 border-blue-100', icon: Target, desc: 'Overall correct predictions of both loyal & churn instances.' },
    { name: 'Precision', val: precision, color: 'text-indigo-600 bg-indigo-50 border-indigo-100', icon: Award, desc: 'Out of all predicted churn targets, how many actually churned.' },
    { name: 'Recall (Sensitivity)', val: recall, color: 'text-rose-600 bg-rose-50 border-rose-100', icon: Percent, desc: 'Out of all actual churned cases, how many did the model intercept.' },
    { name: 'F1-Score', val: f1Score, color: 'text-emerald-600 bg-emerald-50 border-emerald-100', icon: Activity, desc: 'Harmonic balance of Precision & Recall. Ideal for imbalanced data.' }
  ];

  return (
    <div className="space-y-6" id="metrics-root-panel">
      {/* Metrics Score Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {scoreCards.map((sc, idx) => {
          const Icon = sc.icon;
          return (
            <div key={idx} className="bg-white border border-slate-100 rounded-xl p-4 shadow-sm space-y-1.5 hover:border-slate-200 transition-all">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{sc.name}</span>
                <span className={`p-1.5 rounded-lg border ${sc.color}`}>
                  <Icon className="h-4 w-4" />
                </span>
              </div>
              <div className="text-2xl font-bold font-mono text-slate-800">
                {(sc.val * 100).toFixed(1)}%
              </div>
              <p className="text-[10px] text-slate-400 leading-normal">{sc.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Primary Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Confusion Matrix Card */}
        <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-800 mb-1 flex items-center gap-1.5">
              <CheckCircle2 className="h-4 w-4 text-slate-500" />
              Confusion Matrix
            </h3>
            <p className="text-xs text-slate-400 mb-4 pb-2 border-b border-slate-50">
              Evaluates model prediction matches against ground truths.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-1 text-center font-mono text-xs">
            {/* Headers */}
            <div className="p-1" />
            <div className="bg-slate-50 p-2 font-semibold text-[10px] text-slate-500 rounded uppercase">Pred Loyal</div>
            <div className="bg-slate-50 p-2 font-semibold text-[10px] text-slate-500 rounded uppercase">Pred Churn</div>

            {/* Row 1: Actual Loyal */}
            <div className="bg-slate-50 p-2 font-semibold text-[10px] text-slate-500 flex items-center justify-center rounded uppercase">Actual Loyal</div>
            <div className="bg-emerald-50 text-emerald-900 duration-200 p-4 border border-emerald-100 rounded flex flex-col items-center justify-center">
              <span className="text-lg font-extrabold">{tn}</span>
              <span className="text-[9px] text-emerald-600 font-semibold font-sans mt-0.5">True Neg (TN)</span>
            </div>
            <div className="bg-slate-50 text-slate-500 duration-200 p-4 border border-slate-100 rounded flex flex-col items-center justify-center">
              <span className="text-lg font-extrabold">{fp}</span>
              <span className="text-[9px] text-slate-400 font-semibold font-sans mt-0.5">False Pos (FP)</span>
            </div>

            {/* Row 2: Actual Churn */}
            <div className="bg-slate-50 p-2 font-semibold text-[10px] text-slate-500 flex items-center justify-center rounded uppercase">Actual Churn</div>
            <div className="bg-slate-50 text-slate-500 duration-200 p-4 border border-slate-100 rounded flex flex-col items-center justify-center">
              <span className="text-lg font-extrabold">{fn}</span>
              <span className="text-[9px] text-slate-400 font-semibold font-sans mt-0.5">False Neg (FN)</span>
            </div>
            <div className="bg-rose-50 text-rose-900 duration-200 p-4 border border-rose-100 rounded flex flex-col items-center justify-center">
              <span className="text-lg font-bold">{tp}</span>
              <span className="text-[9px] text-rose-600 font-semibold font-sans mt-0.5">True Pos (TP)</span>
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-[10px] text-slate-500 space-y-1 mt-4">
            <p><strong>Note:</strong> Churn analysis prioritizes reducing <strong>False Negatives (FN)</strong> (missing customers who are actually planning to leave).</p>
            <p>• Preprocessing with <strong className="text-amber-600">SMOTE</strong> reduces FN, thereby maximizing <strong>Recall</strong>!</p>
          </div>
        </div>

        {/* ROC Curve Chart Card */}
        <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
                <BarChart3 className="h-4 w-4 text-indigo-500" />
                ROC Crucial Evaluation
              </h3>
              <p className="text-xs text-slate-400">
                Sensitivity (TPR) vs. False Positive Rate (FPR).
              </p>
            </div>
            <div className="text-right">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Est. AUC Score</span>
              <div className="text-sm font-bold font-mono text-indigo-600">{aucScore.toFixed(3)}</div>
            </div>
          </div>

          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics.rocCurve} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
                <defs>
                  <linearGradient id="colorAuc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis 
                  dataKey="fpr" 
                  type="number" 
                  domain={[0, 1]} 
                  tick={{ fontSize: 9, fill: '#94a3b8' }}
                  label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 9 }} 
                />
                <YAxis 
                  dataKey="tpr" 
                  type="number" 
                  domain={[0, 1]} 
                  tick={{ fontSize: 9, fill: '#94a3b8' }}
                  label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 15, fill: '#64748b', fontSize: 9 }} 
                />
                <Tooltip 
                  formatter={(value) => [value, 'Rate']}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-800 text-white p-2 rounded text-[10px] space-y-0.5 shadow">
                          <p className="font-bold">Threshold Step: {data.threshold}</p>
                          <p>TPR (Recall): {data.tpr}</p>
                          <p>FPR (FP Rate): {data.fpr}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                {/* Diagonal random baseline */}
                <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#cbd5e1" strokeDasharray="4 4" dot={false} strokeWidth={1} />
                <Area type="monotone" dataKey="tpr" stroke="none" fillOpacity={1} fill="url(#colorAuc)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 border-t border-slate-50 pt-2">
            <span>An AUC of 1.0 is a perfect model.</span>
            <span>AUC of 0.5 is completely random.</span>
          </div>
        </div>

        {/* Feature Importance Card */}
        <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 mb-1 flex items-center gap-1.5">
            <ListOrdered className="h-4 w-4 text-emerald-500" />
            Class Feature Coefficients
          </h3>
          <p className="text-xs text-slate-400 mb-4 pb-2 border-b border-slate-50">
            Relative influence magnitudes of each variable.
          </p>

          <div className="h-[210px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance} layout="vertical" margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 9, fill: '#94a3b8' }} domain={[0, 1]} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 9, fill: '#475569' }} width={120} />
                <Tooltip 
                  formatter={(value: any) => [Number(value).toFixed(3), 'Importance Weight']}
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '10px' }}
                />
                <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]}>
                  {featureImportance.map((entry, index) => {
                    // Pick distinct colors for top vs bottom
                    const isTop = index < 3;
                    return <Cell key={`cell-${index}`} fill={isTop ? '#5f5af6' : '#94a3b8'} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Epochs Training Log Curves (Logistic & Neural Network only) */}
      {logs.length > 0 && (
        <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-800">Gradient Training Convergence</h3>
              <p className="text-xs text-slate-400">
                Log losses and training accuracy improvements across epoch cycles.
              </p>
            </div>
            <div className="text-xs px-2.5 py-1 bg-slate-100 rounded-full font-bold text-slate-600 uppercase border border-slate-200">
              {modelType === 'logistic_regression' ? 'Gradient Descent' : modelType === 'neural_network' ? 'Backpropagation' : 'Tree Calculations'}
            </div>
          </div>

          <div className="h-[180px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={logs} margin={{ top: 5, right: 15, left: -25, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#94a3b8' }} label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 9 }} />
                <YAxis yAxisId="left" tick={{ fontSize: 9, fill: '#e11d48' }} label={{ value: 'Cross Entropy Loss', angle: -90, position: 'insideLeft', offset: 15, fill: '#e11d48', fontSize: 8 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: '#10b981' }} label={{ value: 'Accuracy', angle: 90, position: 'insideRight', offset: 15, fill: '#10b981', fontSize: 8 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '10px' }} />
                <Legend verticalAlign="top" height={30} iconSize={10} wrapperStyle={{ fontSize: '10px' }} />
                <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2.5} name="Training Loss" activeDot={{ r: 4 }} dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} name="Accuracy" activeDot={{ r: 4 }} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
