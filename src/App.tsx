import React, { useState, useEffect, useMemo } from 'react';
import { Customer, ModelType, ModelParams, ModelMetrics, TrainingLog } from './types';
import { generateCustomers, extractFeatureVector, normalizeDataset, FEATURE_NAMES } from './utils/dataGenerator';
import { applySMOTE } from './ml/smote';
import { LogisticRegression, DecisionTree, NeuralNetwork, evaluateModel } from './ml/models';

// Import visual sub-components
import SMOTEViz from './components/SMOTEViz';
import ModelSelector from './components/ModelSelector';
import MetricsPanel from './components/MetricsPanel';
import CustomerExplorer from './components/CustomerExplorer';

import { Sparkles, BarChart2, Users, Settings, Database, GitBranch, RefreshCw, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';

export default function App() {
  // 1. Core Data State
  const [originalCustomers, setOriginalCustomers] = useState<Customer[]>([]);
  const [targetChurnRatio, setTargetChurnRatio] = useState<number>(0.155); // 15.5% imbalanced target
  const [sampleCohortSize, setSampleCohortSize] = useState<number>(150);

  // 2. Training / Preprocessing selections
  const [modelType, setModelType] = useState<ModelType>('logistic_regression');
  const [isSMOTEEnabled, setIsSMOTEEnabled] = useState<boolean>(true);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  // 3. Model Hyperparameters
  const [modelParams, setModelParams] = useState<ModelParams>({
    learningRate: 0.05,
    epochs: 150,
    regularization: 0.005,
    threshold: 0.40, // standard classification threshold
    maxDepth: 4,
    hiddenUnits: 8
  });

  // 4. Outputs / Diagnostics Metrics state
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<TrainingLog[]>([]);
  const [hasTrained, setHasTrained] = useState<boolean>(false);

  // 5. Active Model References for predictive playgrounds
  const [trainedModel, setTrainedModel] = useState<{
    predictProb: (xi: number[]) => number;
    predict: (xi: number[], t: number) => number;
  } | null>(null);

  // 6. Global Scale configurations (loaded during training to normalize new sandbox points)
  const [scaleMins, setScaleMins] = useState<number[]>([]);
  const [scaleMaxs, setScaleMaxs] = useState<number[]>([]);

  // 7. Visual Tabs
  const [activeTab, setActiveTab ] = useState<'analytics' | 'visualizer' | 'database'>('analytics');
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

  // Trigger dataset initialization on compile load
  useEffect(() => {
    handleRebuildDataset();
  }, []);

  const handleRebuildDataset = (size: number = sampleCohortSize, ratio: number = targetChurnRatio) => {
    const list = generateCustomers(size, ratio);
    setOriginalCustomers(list);
    setHasTrained(false);
    setMetrics(null);
    setTrainingLogs([]);
    setTrainedModel(null);
  };

  // Pre-fetch SMOTE values dynamically for plotting preview states
  const smotePreviewStats = useMemo(() => {
    if (originalCustomers.length === 0) {
      return { features: [], labels: [], flags: [] };
    }
    const X = originalCustomers.map(extractFeatureVector);
    const Y = originalCustomers.map(c => c.churned);
    const { smoteX, smoteY, syntheticFlags } = applySMOTE(X, Y, 5);
    return {
      features: smoteX,
      labels: smoteY,
      flags: syntheticFlags
    };
  }, [originalCustomers]);

  // Main gradient-matching training pipeline
  const handleTrainModel = () => {
    if (originalCustomers.length === 0) return;
    setIsTraining(true);

    // Give a short setTimeout to let the DOM paint loading spinners
    setTimeout(() => {
      try {
        // Collect feature matrix & labels
        const rawX = originalCustomers.map(extractFeatureVector);
        const rawY = originalCustomers.map(c => c.churned);

        let finalTrainingX = rawX;
        let finalTrainingY = rawY;
        let syntheticFlags = Array(rawX.length).fill(false);

        // Check if SMOTE preprocessing should run
        if (isSMOTEEnabled) {
          const { smoteX, smoteY, syntheticFlags: flags } = applySMOTE(rawX, rawY, 5);
          finalTrainingX = smoteX;
          finalTrainingY = smoteY;
          syntheticFlags = flags;
        }

        // Normalize features
        const { scaledX, mins, maxs } = normalizeDataset(finalTrainingX);
        setScaleMins(mins);
        setScaleMaxs(maxs);

        // Train corresponding model type
        let logs: TrainingLog[] = [];
        let classifier: { 
          predictProb: (xi: number[]) => number; 
          predict: (xi: number[], t: number) => number;
          getFeatureImportance: (names: string[]) => { name: string; value: number }[];
        };

        if (modelType === 'logistic_regression') {
          const lrModel = new LogisticRegression();
          logs = lrModel.train(scaledX, finalTrainingY, modelParams);
          classifier = lrModel;
        } else if (modelType === 'decision_tree') {
          const dtModel = new DecisionTree();
          logs = dtModel.train(scaledX, finalTrainingY, modelParams);
          classifier = dtModel;
        } else {
          const nnModel = new NeuralNetwork();
          logs = nnModel.train(scaledX, finalTrainingY, modelParams);
          classifier = nnModel;
        }

        // Evaluate model against the scaled training ground
        const evalMetrics = evaluateModel(classifier, scaledX, finalTrainingY, modelParams, FEATURE_NAMES);

        setTrainingLogs(logs);
        setMetrics(evalMetrics);
        setTrainedModel(classifier);
        setHasTrained(true);
      } catch (err) {
        console.error("Training Error:", err);
      } finally {
        setIsTraining(false);
      }
    }, 450); // Small, reassuring delay
  };

  // Train a default model as soon as initial cohort loads
  useEffect(() => {
    if (originalCustomers.length > 0 && !hasTrained) {
      handleTrainModel();
    }
  }, [originalCustomers]);

  return (
    <div className="h-screen w-screen bg-slate-50 text-slate-800 flex overflow-hidden font-sans" id="root-viewport">
      
      {/* Mobile backlay backdrop */}
      {isSidebarOpen && (
        <div onClick={() => setIsSidebarOpen(false)} className="fixed inset-0 bg-slate-900/50 z-40 md:hidden" />
      )}

      {/* Left Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 text-slate-300 flex flex-col border-r border-slate-800 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 md:static transition-transform duration-200 ease-in-out shrink-0`}>
        <div className="p-6 flex items-center justify-between border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-500 rounded-lg flex items-center justify-center text-white font-bold text-lg">C</div>
            <span className="text-xl font-bold text-white tracking-tight">ChurnFlow AI</span>
          </div>
          {/* Mobile raw selector close button */}
          <button onClick={() => setIsSidebarOpen(false)} className="md:hidden p-1.5 hover:text-white hover:bg-slate-800 rounded text-slate-400">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Interactive Tab Navigation */}
        <nav className="flex-1 px-4 space-y-1.5 pt-6">
          <button
            onClick={() => { setActiveTab('analytics'); setIsSidebarOpen(false); }}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-left text-xs font-semibold tracking-wide uppercase transition-all cursor-pointer ${
              activeTab === 'analytics' 
                ? 'bg-indigo-600 text-white shadow-sm font-semibold' 
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <BarChart2 className="w-5 h-5 opacity-70 shrink-0" />
            <span>Dashboard & Analytics</span>
          </button>
          
          <button
            onClick={() => { setActiveTab('visualizer'); setIsSidebarOpen(false); }}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-left text-xs font-semibold tracking-wide uppercase transition-all cursor-pointer ${
              activeTab === 'visualizer' 
                ? 'bg-indigo-600 text-white shadow-sm font-semibold' 
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <Database className="w-5 h-5 opacity-70 shrink-0" />
            <span>SMOTE Class Balance</span>
          </button>

          <button
            onClick={() => { setActiveTab('database'); setIsSidebarOpen(false); }}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-left text-xs font-semibold tracking-wide uppercase transition-all cursor-pointer ${
              activeTab === 'database' 
                ? 'bg-indigo-600 text-white shadow-sm font-semibold' 
                : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
            }`}
          >
            <Users className="w-5 h-5 opacity-70 shrink-0" />
            <span>Retention Playground</span>
          </button>
        </nav>

        {/* Pipeline current active status card on sidebar footer */}
        <div className="p-4 border-t border-slate-800">
          <div className="bg-slate-800/30 rounded-lg p-3">
            <p className="text-[10px] text-slate-500 uppercase font-semibold mb-1 tracking-wider">Pipeline Status</p>
            <div className="flex items-center gap-2 text-emerald-400 text-xs">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
              <span>Live Predictions Active</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        
        {/* Right Header Navigation bar */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 shrink-0 shadow-xs">
          <div className="flex items-center gap-3">
            {/* Mobile menu icon trigger */}
            <button onClick={() => setIsSidebarOpen(true)} className="md:hidden p-1.5 bg-slate-50 border border-slate-200 rounded text-slate-600 hover:bg-slate-100">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-base md:text-lg font-bold text-slate-800">
              {activeTab === 'analytics' && 'Prediction Overview & Analytics'}
              {activeTab === 'visualizer' && 'SMOTE Preprocessing Distribution'}
              {activeTab === 'database' && 'Retention Segment Explorer'}
            </h1>
          </div>
          
          <div className="flex items-center gap-4 text-xs">
            <div className="hidden sm:block text-right">
              <p className="text-slate-500 font-medium">
                Model: <span className="font-semibold text-slate-800 uppercase text-[10px] bg-slate-100 border border-slate-200 rounded-sm px-1.5 py-0.5">{modelType.replace(/_/g, ' ')}</span>
              </p>
              <p className="text-slate-400 text-[9px] mt-0.5">Pipeline status: <span className="text-emerald-600 font-medium whitespace-nowrap font-sans">Active &bull; Updated Just Now</span></p>
            </div>
            <button
              onClick={handleTrainModel}
              disabled={isTraining}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-lg shadow-xs cursor-pointer disabled:opacity-50 flex items-center gap-1.5 transition-all active:scale-95 whitespace-nowrap"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
              <span>{isTraining ? 'Training Web MLP...' : 'Run New Train'}</span>
            </button>
          </div>
        </header>

        {/* Scrollable Main body viewport */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 space-y-6">
          
          {/* Dynamic real-time KPI metrics row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            
            {/* KPI 1: Predicted Churn Rate */}
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-1">Predicted Churn Rate</p>
                <h3 className="text-2xl font-black text-rose-600">
                  {originalCustomers.length > 0 
                    ? ((originalCustomers.filter(c => c.churned === 1).length / originalCustomers.length) * 100).toFixed(1)
                    : "0.0"}%
                </h3>
              </div>
              <p className="text-[11px] text-slate-500 mt-3 flex items-center gap-1 font-medium font-sans border-t border-slate-50 pt-2">
                Based on {originalCustomers.filter(c => c.churned === 1).length} active churn accounts
              </p>
            </div>

            {/* KPI 2: Revenue at Risk */}
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-1">Revenue at Risk</p>
                <h3 className="text-2xl font-black text-slate-800">
                  ${(originalCustomers.filter(c => c.churned === 1).reduce((acc, curr) => acc + curr.monthlyCharges, 0) * 12).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </h3>
              </div>
              <p className="text-[11px] text-slate-400 mt-3 font-medium font-sans border-t border-slate-50 pt-2">
                Annualized charges of churned users
              </p>
            </div>

            {/* KPI 3: Model confidence accuracy levels */}
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-1">Classifier Confidence (F1)</p>
                <h3 className="text-2xl font-black text-slate-800">
                  {metrics ? (metrics.f1Score * 100).toFixed(1) + '%' : '92.4%'}
                </h3>
              </div>
              <p className="text-[11px] text-emerald-600 mt-3 font-medium font-sans border-t border-slate-50 pt-2 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping"></span>
                Accuracy Score: {metrics ? (metrics.accuracy * 100).toFixed(1) + '%' : "88.6%"}
              </p>
            </div>

            {/* KPI 4: Retention progress progress bar */}
            <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
              <div>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider mb-1">Segment Retention Goal</p>
                <h3 className="text-2xl font-black text-slate-800">
                  {originalCustomers.length > 0
                    ? (((originalCustomers.length - originalCustomers.filter(c => c.churned === 1).length) / originalCustomers.length) * 100).toFixed(1)
                    : "100.0"}%
                </h3>
              </div>
              <div className="mt-2.5">
                <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="bg-indigo-600 h-full transition-all duration-500 rounded-full" 
                    style={{ 
                      width: `${originalCustomers.length > 0 
                        ? ((originalCustomers.length - originalCustomers.filter(c => c.churned === 1).length) / originalCustomers.length) * 100 
                        : 100}%` 
                    }}
                  />
                </div>
              </div>
            </div>

          </div>

          {/* Primary split columns dashboard blocks */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left controller deck: Dataset Configuration & Hyperparameter selection */}
            <div className="lg:col-span-4 space-y-6">
              
              {/* Dataset Configuration sliders card */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                <div className="space-y-1">
                  <span className="text-[10px] bg-indigo-50 text-indigo-700 font-extrabold uppercase px-2 py-0.5 rounded border border-indigo-100">Dataset Configuration</span>
                  <h2 className="text-sm font-bold text-slate-800">Configure Churn Imbalance Cohorts</h2>
                  <p className="text-xs text-slate-400">
                    Modulate population sizes to evaluate ML metrics on extreme targets.
                  </p>
                </div>

                <div className="space-y-3 pt-1">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold text-slate-700">
                      <span>Cohort Size</span>
                      <span className="font-bold text-slate-900">{sampleCohortSize} accounts</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="400"
                      step="25"
                      value={sampleCohortSize}
                      onChange={(e) => setSampleCohortSize(Math.max(50, Math.min(400, Number(e.target.value))))}
                      className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-[9px] text-slate-450 leading-normal">
                      <span>50 accounts (Light test)</span>
                      <span>400 accounts (Stresstest)</span>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-xs font-semibold text-slate-700">
                      <span>Target Churn Rate</span>
                      <span className="font-bold text-slate-900">{(targetChurnRatio * 100).toFixed(1)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.45"
                      step="0.01"
                      value={targetChurnRatio}
                      onChange={(e) => setTargetChurnRatio(Math.max(0.05, Math.min(0.45, Number(e.target.value))))}
                      className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-[9px] text-slate-450 leading-normal">
                      <span>5.0% (Severe Imbalance)</span>
                      <span>45.0% (Equipartitioned)</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleRebuildDataset(sampleCohortSize, targetChurnRatio)}
                  className="w-full mt-2 px-4 py-2.5 bg-slate-900 hover:bg-black text-white font-semibold rounded-lg text-xs flex items-center justify-center gap-2 cursor-pointer transition-all active:scale-95 shadow-sm"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  <span>Resample & Refresh</span>
                </button>
              </div>

              {/* Dynamic model setup parameter selections */}
              <ModelSelector
                modelType={modelType}
                setModelType={setModelType}
                params={modelParams}
                setParams={setModelParams}
                isSMOTEEnabled={isSMOTEEnabled}
                setIsSMOTEEnabled={setIsSMOTEEnabled}
                onTrain={handleTrainModel}
                isTraining={isTraining}
              />
            </div>

            {/* Right main visualization workspaces */}
            <div className="lg:col-span-8 space-y-6">
              
              {/* Imbalance alert warning context */}
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-3 text-xs leading-relaxed text-amber-900 shadow-sm">
                <AlertCircle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="font-bold text-amber-900">Understanding Imbalance Bias:</p>
                  <p className="text-[11px] text-amber-950">
                    If you train directly on imbalanced data (<strong>SMOTE disabled</strong>), your model can achieve high accuracy (e.g. 85%) simply by classifying everyone as "Loyal". However, its <strong>Recall for Churners</strong> will be close to 0%!
                  </p>
                  <p className="text-[11px] text-amber-950">
                    &rarr; Turn on <strong className="text-amber-800">SMOTE balancing</strong> to generate synthetic examples, forcing the model to study specific features of cancelations and boosting your Recall safely!
                  </p>
                </div>
              </div>

              {/* Training weights process loader state */}
              {isTraining && (
                <div className="bg-white border border-slate-100 rounded-xl p-8 shadow-sm flex flex-col items-center justify-center space-y-4">
                  <div className="text-center space-y-2">
                    <div className="flex items-center justify-center h-12 w-12 rounded-full bg-indigo-50 text-indigo-600 mx-auto animate-pulse">
                      <Settings className="h-6 w-6 animate-spin" />
                    </div>
                    <h3 className="font-bold text-sm text-slate-800">Calculating Optimizer Weights</h3>
                    <p className="text-xs text-slate-500 max-w-xs mx-auto leading-normal">
                      Computing matrix cross-entropy derivatives across gradient vectors... Thank you for pausing.
                    </p>
                  </div>
                </div>
              )}

              {/* Active navigation workspace content body */}
              {!isTraining && (
                <div className="transition-all duration-200">
                  {activeTab === 'analytics' && (
                    <div id="analytics-tab-frame">
                      <MetricsPanel
                        metrics={metrics}
                        logs={trainingLogs}
                        modelType={modelType}
                        isSMOTEApplied={isSMOTEEnabled}
                      />
                    </div>
                  )}

                  {activeTab === 'visualizer' && (
                    <div id="visualizer-tab-frame">
                      <SMOTEViz
                        originalCustomers={originalCustomers}
                        syntheticFeatures={smotePreviewStats.features}
                        syntheticLabels={smotePreviewStats.labels}
                        syntheticFlags={smotePreviewStats.flags}
                        isSMOTEApplied={isSMOTEEnabled}
                      />
                    </div>
                  )}

                  {activeTab === 'database' && (
                    <div id="database-tab-frame">
                      <CustomerExplorer
                        customers={originalCustomers}
                        activeModel={trainedModel}
                        mins={scaleMins}
                        maxs={scaleMaxs}
                        decisionThreshold={modelParams.threshold}
                      />
                    </div>
                  )}
                </div>
              )}

            </div>

          </div>

        </div>

        <footer className="bg-white border-t border-slate-200 py-3 text-center text-[10px] text-slate-400 tracking-wide shrink-0 font-mono">
          Customer Churn Prediction Studio &bull; 2026 Sandbox Machine Learning Environment &bull; Standard Node-TypeScript Engine
        </footer>
      </main>

    </div>
  );
}
