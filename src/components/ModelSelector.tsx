import React from 'react';
import { ModelType, ModelParams } from '../types';
import { Sliders, ToggleLeft, ToggleRight, Sparkles, BrainCircuit, GitCommit, Settings, HelpCircle } from 'lucide-react';

interface ModelSelectorProps {
  modelType: ModelType;
  setModelType: (type: ModelType) => void;
  params: ModelParams;
  setParams: (p: ModelParams) => void;
  isSMOTEEnabled: boolean;
  setIsSMOTEEnabled: (v: boolean) => void;
  onTrain: () => void;
  isTraining: boolean;
}

export default function ModelSelector({
  modelType,
  setModelType,
  params,
  setParams,
  isSMOTEEnabled,
  setIsSMOTEEnabled,
  onTrain,
  isTraining
}: ModelSelectorProps) {

  const handleParamChange = (key: keyof ModelParams, val: number) => {
    setParams({
      ...params,
      [key]: val
    });
  };

  const getModelIcon = (type: ModelType) => {
    switch (type) {
      case 'logistic_regression':
        return <GitCommit className="h-5 w-5 text-indigo-500" />;
      case 'decision_tree':
        return <Sliders className="h-5 w-5 text-emerald-500" />;
      case 'neural_network':
        return <BrainCircuit className="h-5 w-5 text-purple-500" />;
    }
  };

  return (
    <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm space-y-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="p-1.5 bg-indigo-50 rounded-lg text-indigo-600">
            <Settings className="h-5 w-5" />
          </span>
          <h3 className="text-lg font-semibold text-slate-800">Model Configuration</h3>
        </div>
        <p className="text-xs text-slate-500">
          Configure parameters and preprocessing pipelines for churn classification.
        </p>
      </div>

      {/* Model Selection Tabs */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Select Algorithm</label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
          {[
            { id: 'logistic_regression', title: 'Logistic Regression', desc: 'Predictive probability curve using L2 gradient step descent' },
            { id: 'decision_tree', title: 'Decision Tree', desc: 'Binary splitting tree based on Gini entropy reductions' },
            { id: 'neural_network', title: 'Neural Network (MLP)', desc: 'Multi-layer perceptron with backpropagation backsteps' }
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => setModelType(m.id as ModelType)}
              disabled={isTraining}
              className={`flex flex-col items-start p-3 rounded-xl border text-left transition-all duration-200 cursor-pointer ${
                modelType === m.id
                  ? 'border-indigo-600 bg-indigo-50/20 shadow-sm'
                  : 'border-slate-100 hover:bg-slate-50 hover:border-slate-200'
              } disabled:opacity-50`}
            >
              <div className="flex items-center gap-2 mb-1.5">
                {getModelIcon(m.id as ModelType)}
                <span className="text-sm font-semibold text-slate-800">{m.title}</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-normal">{m.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Preprocessing Pipelines (SMOTE) */}
      <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-slate-800 font-semibold text-sm">
            <span>Preprocess with SMOTE Balancing</span>
            <span className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0.5 rounded-full font-bold">Recommended</span>
          </div>
          <p className="text-xs text-slate-500 leading-normal max-w-md">
            Synthesize fake minority customer churn cases to balance testing weights. Fixes model bias against non-churn predictions.
          </p>
        </div>
        <button
          onClick={() => setIsSMOTEEnabled(!isSMOTEEnabled)}
          disabled={isTraining}
          className={`flex items-center justify-between gap-2 px-4 py-2.5 rounded-xl border font-semibold text-xs transition-all cursor-pointer ${
            isSMOTEEnabled 
              ? 'bg-amber-500 border-amber-600 text-white hover:bg-amber-600 shadow-sm' 
              : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
          } disabled:opacity-50`}
        >
          {isSMOTEEnabled ? <ToggleRight className="h-5 w-5" /> : <ToggleLeft className="h-5 w-5" />}
          <span>{isSMOTEEnabled ? 'SMOTE Balanced' : 'Raw Imbalanced'}</span>
        </button>
      </div>

      {/* Hyperparameters Sliders */}
      <div className="space-y-4">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Hyperparameters</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Decision Threshold slider (universal) */}
          <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
            <div className="flex justify-between text-xs font-medium text-slate-700">
              <span className="flex items-center gap-1">Classification Threshold <HelpCircle className="h-3 w-3 text-slate-400" title="Probability required to classify as churned" /></span>
              <span className="font-bold text-slate-900">{params.threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={params.threshold}
              onChange={(e) => handleParamChange('threshold', Number(e.target.value))}
              disabled={isTraining}
              className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>0.10 (High Recall)</span>
              <span>0.90 (High Precision)</span>
            </div>
          </div>

          {/* LR / Epochs for Gradient descent models */}
          {modelType !== 'decision_tree' && (
            <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>Learning Rate (Alpha)</span>
                <span className="font-bold text-slate-900">{params.learningRate.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.005"
                max="0.3"
                step="0.005"
                value={params.learningRate}
                onChange={(e) => handleParamChange('learningRate', Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>0.005 (Secure)</span>
                <span>0.30 (Aggressive)</span>
              </div>
            </div>
          )}

          {modelType !== 'decision_tree' && (
            <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>Max Search Epochs</span>
                <span className="font-bold text-slate-900">{params.epochs}</span>
              </div>
              <input
                type="range"
                min="10"
                max="300"
                step="10"
                value={params.epochs}
                onChange={(e) => handleParamChange('epochs', Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>10 iterations</span>
                <span>300 iterations</span>
              </div>
            </div>
          )}

          {/* L2 Regularization (Logistic Regression only) */}
          {modelType === 'logistic_regression' && (
            <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>L2 Regularization Coefficient</span>
                <span className="font-bold text-slate-900">{params.regularization.toFixed(4)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.05"
                step="0.001"
                value={params.regularization}
                onChange={(e) => handleParamChange('regularization', Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>0.000 (No penalty)</span>
                <span>0.050 (High decay)</span>
              </div>
            </div>
          )}

          {/* Max Depth (Decision Tree only) */}
          {modelType === 'decision_tree' && (
            <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>Maximum Tree Depth</span>
                <span className="font-bold text-slate-900">{params.maxDepth}</span>
              </div>
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={params.maxDepth}
                onChange={(e) => handleParamChange('maxDepth', Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>2 (Weak stumps)</span>
                <span>8 (Possible Overfit)</span>
              </div>
            </div>
          )}

          {/* Hidden units count (Neural network only) */}
          {modelType === 'neural_network' && (
            <div className="space-y-1.5 bg-slate-50/50 p-3 rounded-lg border border-slate-100">
              <div className="flex justify-between text-xs font-medium text-slate-700">
                <span>Hidden Units (Dense layer)</span>
                <span className="font-bold text-slate-900">{params.hiddenUnits} neurons</span>
              </div>
              <input
                type="range"
                min="4"
                max="16"
                step="1"
                value={params.hiddenUnits}
                onChange={(e) => handleParamChange('hiddenUnits', Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-indigo-600 h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>4 units</span>
                <span>16 units</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Train button */}
      <div className="pt-2">
        <button
          onClick={onTrain}
          disabled={isTraining}
          className="w-full py-3 bg-indigo-600 text-white rounded-xl shadow-md hover:bg-indigo-700 active:scale-95 transition-all duration-150 flex items-center justify-center gap-2 font-semibold text-sm cursor-pointer disabled:opacity-50 disabled:pointer-events-none"
        >
          {isTraining ? (
            <>
              <Sparkles className="h-5 w-5 animate-spin" />
              <span>Optimizing Weight Gradients...</span>
            </>
          ) : (
            <>
              <BrainCircuit className="h-5 w-5" />
              <span>Train Churn Classifier</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
