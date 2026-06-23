import React, { useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceArea } from 'recharts';
import { Customer } from '../types';
import { HelpCircle, RefreshCw, Layers } from 'lucide-react';

interface SMOTEVizProps {
  originalCustomers: Customer[];
  syntheticFeatures: number[][]; // full features list after SMOTE
  syntheticLabels: number[];     // labels after SMOTE
  syntheticFlags: boolean[];     // flags identifying if synthetic
  isSMOTEApplied: boolean;
}

export default function SMOTEViz({
  originalCustomers,
  syntheticFeatures,
  syntheticLabels,
  syntheticFlags,
  isSMOTEApplied
}: SMOTEVizProps) {
  const [xAxisFeature, setXAxisFeature] = useState<number>(3); // Support Calls by default
  const [yAxisFeature, setYAxisFeature] = useState<number>(1); // Monthly Charges by default

  const featuresList = [
    { idx: 0, name: 'Tenure (Months)', min: 0, max: 72 },
    { idx: 1, name: 'Monthly Charges ($)', min: 0, max: 150 },
    { idx: 2, name: 'Usage Pattern Factor', min: 0, max: 1.5 },
    { idx: 3, name: 'Support Calls', min: 0, max: 10 },
    { idx: 6, name: 'Billing Issues (0 or 1)', min: 0, max: 1 }
  ];

  // Map dataset for Recharts Scatter plot
  const chartData = useMemo(() => {
    if (!isSMOTEApplied) {
      // Map original customers features
      return originalCustomers.map(c => ({
        x: xAxisFeature === 0 ? c.tenure :
           xAxisFeature === 1 ? c.monthlyCharges :
           xAxisFeature === 2 ? c.usageFactor :
           xAxisFeature === 3 ? c.supportCalls : c.billingIssues,
        y: yAxisFeature === 0 ? c.tenure :
           yAxisFeature === 1 ? c.monthlyCharges :
           yAxisFeature === 2 ? c.usageFactor :
           yAxisFeature === 3 ? c.supportCalls : yAxisFeature === 6 ? c.billingIssues : c.monthlyCharges,
        label: c.churned === 1 ? 'Churned (Original)' : 'Loyal',
        isSynthetic: false,
        name: c.name,
      }));
    } else {
      // Map post-SMOTE features
      return syntheticFeatures.map((feats, idx) => {
        const isSynth = syntheticFlags[idx];
        const isChurnPred = syntheticLabels[idx] === 1;
        const label = isSynth 
          ? 'Synthetic Churned (SMOTE)' 
          : isChurnPred 
            ? 'Churned (Original)' 
            : 'Loyal';

        // Recover unscaled values for display (approximation based on indices mapping)
        const valX = feats[xAxisFeature];
        const valY = feats[yAxisFeature];
        
        // Reverse scale estimation or just show relative normalized value
        return {
          x: Number(valX.toFixed(2)),
          y: Number(valY.toFixed(2)),
          label,
          isSynthetic: isSynth,
          name: isSynth ? `Synthetic Node #${idx}` : `Original Customer #${idx}`,
        };
      });
    }
  }, [originalCustomers, syntheticFeatures, syntheticLabels, syntheticFlags, isSMOTEApplied, xAxisFeature, yAxisFeature]);

  // Statistics about classes
  const counts = useMemo(() => {
    const total = chartData.length;
    const loyal = chartData.filter(d => d.label === 'Loyal').length;
    const originalChurn = chartData.filter(d => d.label === 'Churned (Original)').length;
    const syntheticChurn = chartData.filter(d => d.label === 'Synthetic Churned (SMOTE)').length;
    const churnTotal = originalChurn + syntheticChurn;

    return {
      total,
      loyal,
      loyalPercent: ((loyal / total) * 100).toFixed(1),
      churnTotal,
      churnPercent: ((churnTotal / total) * 100).toFixed(1),
      originalChurn,
      syntheticChurn,
    };
  }, [chartData]);

  const xFeatureInfo = featuresList.find(f => f.idx === xAxisFeature);
  const yFeatureInfo = featuresList.find(f => f.idx === yAxisFeature);

  return (
    <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm">
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 pb-4 border-b border-slate-100 gap-4" id="smote-section-header">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 bg-amber-50 rounded-lg text-amber-600">
              <Layers className="h-5 w-5" />
            </span>
            <h3 className="text-lg font-semibold text-slate-800">SMOTE Class Balance Visualizer</h3>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            See how the data is balance-corrected in real-time. Original customer churn data is heavily imbalanced.
          </p>
        </div>

        {/* Feature Selectors */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">X-Axis:</span>
            <select
              value={xAxisFeature}
              onChange={(e) => setXAxisFeature(Number(e.target.value))}
              className="bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded px-2 py-1 text-slate-600 font-medium focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
            >
              {featuresList.map(f => (
                <option key={f.idx} value={f.idx}>{f.name}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">Y-Axis:</span>
            <select
              value={yAxisFeature}
              onChange={(e) => setYAxisFeature(Number(e.target.value))}
              className="bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded px-2 py-1 text-slate-600 font-medium focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
            >
              {featuresList.map(f => (
                <option key={f.idx} value={f.idx}>{f.name}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Statistics Cards */}
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-slate-50 rounded-xl p-4 space-y-3 border border-slate-100">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Class Distribution</h4>
            
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium text-slate-600">
                <span>Loyal Customers (Majority)</span>
                <span>{counts.loyal} ({counts.loyalPercent}%)</span>
              </div>
              <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-emerald-500 transition-all duration-500" 
                  style={{ width: `${counts.loyalPercent}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium text-slate-600">
                <span>Churned (Before SMOTE)</span>
                <span>{counts.originalChurn}</span>
              </div>
              {isSMOTEApplied && (
                <div className="flex justify-between text-xs font-medium text-amber-600">
                  <span>Synthetic (Added by SMOTE)</span>
                  <span>+{counts.syntheticChurn}</span>
                </div>
              )}
              <div className="flex justify-between text-xs font-semibold text-rose-600 pt-1">
                <span>Total Churn Sample</span>
                <span>{counts.churnTotal} ({counts.churnPercent}%)</span>
              </div>
              <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-rose-500 transition-all duration-500" 
                  style={{ width: `${counts.churnPercent}%` }}
                />
              </div>
            </div>
          </div>

          <div className="p-4 bg-blue-50/50 rounded-xl border border-blue-100 space-y-2">
            <div className="flex items-center gap-1.5 text-blue-700">
              <HelpCircle className="h-4 w-4 shrink-0" />
              <h5 className="text-xs font-semibold">How SMOTE Works</h5>
            </div>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              Standard customer churn databases are <strong>highly imbalanced</strong> (e.g. 85% loyal, 15% churn). Standard ML classifiers struggle, over-predicting the majority class.
            </p>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              <strong>SMOTE</strong> selects a random minority customer (Churn), finds their 5 nearest neighbors in the same class, and interpolates <strong>new synthetic customer profiles</strong> along the line segments. This balances the training grounds without simple replication!
            </p>
          </div>
        </div>

        {/* 2D Scatter plot rendering */}
        <div className="lg:col-span-3">
          <div className="h-[280px] w-full" id="scatter-plot bg-slate-50/10 rounded">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name={xFeatureInfo?.name} 
                  label={{ value: xFeatureInfo?.name, position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  domain={isSMOTEApplied ? [0, 1] : [0, 'auto']}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name={yFeatureInfo?.name} 
                  label={{ value: yFeatureInfo?.name, angle: -90, position: 'insideLeft', offset: 15, fill: '#64748b', fontSize: 11 }}
                  tick={{ fill: '#64748b', fontSize: 10 }}
                  domain={isSMOTEApplied ? [0, 1] : [0, 'auto']}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-800 text-white p-2.5 rounded-lg text-[11px] space-y-1 shadow-lg border border-slate-700 max-w-[200px]">
                          <p className="font-bold border-b border-slate-700 pb-0.5 mb-1">{data.name}</p>
                          <p><span className="text-slate-400">Class:</span> <span className="font-semibold">{data.label}</span></p>
                          <p><span className="text-slate-400">{xFeatureInfo?.name}:</span> {data.x}</p>
                          <p><span className="text-slate-400">{yFeatureInfo?.name}:</span> {data.y}</p>
                          {data.isSynthetic && (
                            <p className="text-amber-400 font-medium">✨ Synthetic sample generated by k-NN interpolation</p>
                          )}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Loyal" data={chartData.filter(d => d.label === 'Loyal')} fill="#10b981" line={false}>
                  {chartData.filter(d => d.label === 'Loyal').map((entry, index) => (
                    <Cell key={`cell-loyal-${index}`} fill="#10b981" fillOpacity={0.6} stroke="#059669" strokeWidth={1} r={5} />
                  ))}
                </Scatter>
                <Scatter name="Churned (Original)" data={chartData.filter(d => d.label === 'Churned (Original)')} fill="#f43f5e" line={false}>
                  {chartData.filter(d => d.label === 'Churned (Original)').map((entry, index) => (
                    <Cell key={`cell-original-${index}`} fill="#f43f5e" fillOpacity={0.8} stroke="#e11d48" strokeWidth={1.5} r={6} />
                  ))}
                </Scatter>
                {isSMOTEApplied && (
                  <Scatter name="Synthetic Churned (SMOTE)" data={chartData.filter(d => d.label === 'Synthetic Churned (SMOTE)')} fill="#f59e0b" line={false}>
                    {chartData.filter(d => d.label === 'Synthetic Churned (SMOTE)').map((entry, index) => (
                      <Cell key={`cell-synthetic-${index}`} fill="#eab308" className="animate-pulse" fillOpacity={0.9} stroke="#ca8a04" strokeWidth={1.5} r={7} />
                    ))}
                  </Scatter>
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap items-center justify-center gap-6 mt-4 text-xs font-semibold text-slate-600">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-emerald-500 rounded-full border border-emerald-600 opacity-80" />
              <span>Loyal (Ground Truth)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-rose-500 rounded-full border border-rose-600" />
              <span>Churned (Ground Truth)</span>
            </div>
            {isSMOTEApplied && (
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-3.5 bg-yellow-400 rounded-full border border-yellow-600 animate-pulse" />
                <span className="text-amber-700">Synthetic Churn Balanced Points (SMOTE)</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
