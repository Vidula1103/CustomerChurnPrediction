import React, { useState, useMemo } from 'react';
import { Customer, ModelMetrics } from '../types';
import { extractFeatureVector, scaleSingleVector, FEATURE_NAMES } from '../utils/dataGenerator';
import { 
  Users, Search, Filter, AlertTriangle, CheckCircle, HelpCircle, 
  ChevronRight, Brain, Sparkles, Server, FileText, BadgeAlert, Send, Lightbulb, Play 
} from 'lucide-react';

interface CustomerExplorerProps {
  customers: Customer[];
  activeModel: {
    predictProb: (xi: number[]) => number;
    predict: (xi: number[], t: number) => number;
  } | null;
  mins: number[]; // feature scaling parameters
  maxs: number[];
  decisionThreshold: number;
}

export default function CustomerExplorer({
  customers,
  activeModel,
  mins,
  maxs,
  decisionThreshold
}: CustomerExplorerProps) {
  // Navigation & Search State
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState<'all' | 'high_risk' | 'support_heavy' | 'month_to_month' | 'misclassified'>('all');
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(customers[0]?.id || null);

  // Custom Sandbox slide parameters
  const [sandboxMode, setSandboxMode] = useState<boolean>(false);
  const [sandboxTenure, setSandboxTenure] = useState(12);
  const [sandboxCharges, setSandboxCharges] = useState(75);
  const [sandboxUsage, setSandboxUsage] = useState(0.8);
  const [sandboxSupport, setSandboxSupport] = useState(2);
  const [sandboxContract, setSandboxContract] = useState<'month-to-month' | 'one-year' | 'two-year'>('month-to-month');
  const [sandboxBilling, setSandboxBilling] = useState(0);

  // AI Playbook Generation state
  const [isGeneratingPlaybook, setIsGeneratingPlaybook] = useState(false);
  const [generatedPlaybook, setGeneratedPlaybook] = useState<string | null>(null);
  const [playbookError, setPlaybookError] = useState<string | null>(null);

  // Map each customer to their live prediction score
  const customersWithPredictions = useMemo(() => {
    return customers.map(c => {
      if (!activeModel || mins.length === 0) {
        return { ...c, predictedProb: c.churned * 0.8 + Math.random() * 0.15, predictedLabel: c.churned };
      }
      const rawVector = extractFeatureVector(c);
      const scaledVector = scaleSingleVector(rawVector, mins, maxs);
      const prob = activeModel.predictProb(scaledVector);
      const pred = activeModel.predict(scaledVector, decisionThreshold);
      return {
        ...c,
        predictedProb: prob,
        predictedLabel: pred
      };
    });
  }, [customers, activeModel, mins, maxs, decisionThreshold]);

  // Compute stats on current batch
  const statistics = useMemo(() => {
    const total = customersWithPredictions.length;
    const modelActive = activeModel !== null;
    let falsePositives = 0;
    let falseNegatives = 0;
    let highRiskCount = 0;

    customersWithPredictions.forEach(c => {
      if (c.predictedProb >= decisionThreshold) highRiskCount++;
      if (c.predictedLabel === 1 && c.churned === 0) falsePositives++;
      if (c.predictedLabel === 0 && c.churned === 1) falseNegatives++;
    });

    return {
      total,
      highRiskCount,
      highRiskPercent: total === 0 ? 0 : Math.round((highRiskCount / total) * 100),
      falsePositives,
      falseNegatives,
      modelActive
    };
  }, [customersWithPredictions, decisionThreshold, activeModel]);

  // Handle filtrations
  const filteredCustomers = useMemo(() => {
    return customersWithPredictions.filter(c => {
      const matchesSearch = c.name.toLowerCase().includes(searchTerm.toLowerCase()) || c.id.toLowerCase().includes(searchTerm.toLowerCase());
      if (!matchesSearch) return false;

      if (riskFilter === 'all') return true;
      if (riskFilter === 'high_risk') return c.predictedProb >= decisionThreshold;
      if (riskFilter === 'support_heavy') return c.supportCalls >= 4;
      if (riskFilter === 'month_to_month') return c.contractType === 'month-to-month';
      if (riskFilter === 'misclassified') {
        return c.predictedLabel !== c.churned;
      }
      return true;
    });
  }, [customersWithPredictions, searchTerm, riskFilter, decisionThreshold]);

  // Active target customer selected for deep-dive / custom slider
  const selectedCustomerInfo = useMemo(() => {
    if (sandboxMode) {
      // Sandbox synthetic mock customer
      const rawVector = [
        sandboxTenure,
        sandboxCharges,
        sandboxUsage,
        sandboxSupport,
        sandboxContract === 'month-to-month' ? 1 : 0,
        sandboxContract === 'one-year' ? 1 : 0,
        sandboxBilling
      ];
      let prob = 0.5;
      let label = 0;

      if (activeModel && mins.length > 0) {
        const scaledVector = scaleSingleVector(rawVector, mins, maxs);
        prob = activeModel.predictProb(scaledVector);
        label = activeModel.predict(scaledVector, decisionThreshold);
      }

      return {
        id: 'SANDBOX-001',
        name: 'Sandbox Test Profile',
        tenure: sandboxTenure,
        monthlyCharges: sandboxCharges,
        usageFactor: sandboxUsage,
        supportCalls: sandboxSupport,
        contractType: sandboxContract,
        billingIssues: sandboxBilling,
        predictedProb: prob,
        predictedLabel: label,
        churned: -1 // Unknown, purely predictive
      };
    } else {
      return customersWithPredictions.find(c => c.id === selectedCustomerId) || null;
    }
  }, [
    selectedCustomerId, sandboxMode, sandboxTenure, sandboxCharges, 
    sandboxUsage, sandboxSupport, sandboxContract, sandboxBilling, 
    customersWithPredictions, activeModel, mins, maxs, decisionThreshold
  ]);

  // Reset or initialize Gemini Outreach playbook
  const handleGeneratePlaybook = async () => {
    if (!selectedCustomerInfo) return;
    setIsGeneratingPlaybook(true);
    setGeneratedPlaybook(null);
    setPlaybookError(null);

    try {
      const response = await fetch("/api/retention-playbook", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedCustomerInfo.name,
          tenure: selectedCustomerInfo.tenure,
          monthlyCharges: selectedCustomerInfo.monthlyCharges,
          usageFactor: selectedCustomerInfo.usageFactor,
          supportCalls: selectedCustomerInfo.supportCalls,
          contractType: selectedCustomerInfo.contractType,
          billingIssues: selectedCustomerInfo.billingIssues,
          predictedRisk: selectedCustomerInfo.predictedProb
        }),
      });

      const data = await response.json();
      if (data.success) {
        setGeneratedPlaybook(data.playbook);
      } else {
        setPlaybookError(data.error || "Failed to parse Response from Server proxy.");
      }
    } catch (err: any) {
      setPlaybookError(err.message || "Could not complete network request to fullstack server.");
    } finally {
      setIsGeneratingPlaybook(false);
    }
  };

  const getRiskBadgeColor = (prob: number) => {
    if (prob < 0.3) return 'bg-emerald-50 text-emerald-700 border-emerald-100';
    if (prob < 0.6) return 'bg-amber-50 text-amber-700 border-amber-100';
    return 'bg-rose-50 text-rose-700 border-rose-100 animate-pulse';
  };

  const getRiskExplanation = (prob: number) => {
    if (prob < 0.3) return 'Low relative churn probability. Account shows high stability factors.';
    if (prob < 0.6) return 'Moderate concern. Minor risk signatures detected, suggest routine engagement.';
    return 'CRITICAL THREAT. High likelihood of cancelation within next billing cycle. Needs immediate intervention!';
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6" id="customer-explorer-section">
      
      {/* LHS: Customer list and filters (7 cols on large desktop) */}
      <div className="xl:col-span-7 bg-white border border-slate-100 rounded-xl p-5 shadow-sm space-y-4">
        
        {/* Module Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-50 gap-3">
          <div>
            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-1.5">
              <Users className="h-5 w-5 text-indigo-500" />
              Customer Database Analyzer
            </h3>
            <p className="text-xs text-slate-500">
              Browse historical active batch, check model predictions, and evaluate risk tiers.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setSandboxMode(false);
                setGeneratedPlaybook(null);
              }}
              className={`px-3 py-1.5 rounded-lg border text-xs font-semibold cursor-pointer transition-all ${
                !sandboxMode 
                  ? 'bg-indigo-600 border-indigo-700 text-white shadow-sm' 
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              Real Customers
            </button>
            <button
              onClick={() => {
                setSandboxMode(true);
                setGeneratedPlaybook(null);
              }}
              className={`px-3 py-1.5 rounded-lg border text-xs font-semibold cursor-pointer transition-all flex items-center gap-1 ${
                sandboxMode 
                  ? 'bg-indigo-600 border-indigo-700 text-white shadow-sm' 
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              <Lightbulb className="h-3.5 w-3.5" />
              Sandbox Simulator
            </button>
          </div>
        </div>

        {!sandboxMode ? (
          <>
            {/* Search & Filtration control deck */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Search input */}
              <div className="relative">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Query customer name or ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-slate-50/50 hover:bg-slate-50 border border-slate-200 rounded-xl text-slate-700 placeholder-slate-400 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white transition-all"
                />
              </div>

              {/* Advanced Filter selects */}
              <div className="relative">
                <Filter className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                <select
                  value={riskFilter}
                  onChange={(e: any) => setRiskFilter(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-slate-50/50 hover:bg-slate-50 border border-slate-200 rounded-xl text-slate-700 font-medium text-xs focus:outline-none cursor-pointer focus:ring-1 focus:ring-indigo-500 transition-all appearance-none"
                >
                  <option value="all">Filter: Show All Customers</option>
                  <option value="high_risk">Filter: Predicted Churn (High Risk)</option>
                  <option value="support_heavy">Filter: Critical Support cases (&ge; 4 calls)</option>
                  <option value="month_to_month">Filter: Contract: Month-to-month</option>
                  <option value="misclassified">Filter: Model Misclassifications (Debug Mode)</option>
                </select>
              </div>
            </div>

            {/* Quick Insights pill bar */}
            <div className="grid grid-cols-3 gap-2 text-center bg-slate-50 p-2.5 rounded-xl border border-slate-100 text-[11px] font-semibold text-slate-600">
              <div>
                <span>Database Batch</span>
                <p className="text-xs font-bold text-slate-800 font-mono mt-0.5">{statistics.total}</p>
              </div>
              <div className="border-x border-slate-200/60">
                <span>Total Predicted Churn</span>
                <p className="text-xs font-bold text-rose-600 font-mono mt-0.5">{statistics.highRiskCount} ({statistics.highRiskPercent}%)</p>
              </div>
              <div>
                <span>Model Divergences</span>
                <p className="text-xs font-bold text-amber-600 font-mono mt-0.5">
                  FP: {statistics.falsePositives} | FN: {statistics.falseNegatives}
                </p>
              </div>
            </div>

            {/* Database Interactive Table */}
            <div className="overflow-x-auto max-h-[360px] border border-slate-100 rounded-xl">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50/70 border-b border-slate-100 text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                    <th className="p-3">Customer Profile</th>
                    <th className="p-3">Usage & Comps</th>
                    <th className="p-3">Contract Style</th>
                    <th className="p-3">Ground Truth</th>
                    <th className="p-3">Model Churn Risk</th>
                    <th className="p-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {filteredCustomers.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="text-center p-8 text-slate-400">
                        No matches found for search criteria. Try modifying parameters.
                      </td>
                    </tr>
                  ) : (
                    filteredCustomers.map(c => {
                      const isSelected = selectedCustomerId === c.id;
                      return (
                        <tr 
                          key={c.id} 
                          onClick={() => {
                            setSelectedCustomerId(c.id);
                            setGeneratedPlaybook(null);
                          }}
                          className={`hover:bg-slate-50/70 cursor-pointer transition-all ${
                            isSelected ? 'bg-indigo-50/30' : ''
                          }`}
                        >
                          <td className="p-3">
                            <div className="font-semibold text-slate-800">{c.name}</div>
                            <div className="text-[10px] font-mono text-slate-400">{c.id} • Tenure: {c.tenure}m</div>
                          </td>
                          <td className="p-3">
                            <div>Support Cases: <strong className={c.supportCalls >= 4 ? 'text-rose-600 font-extrabold' : 'text-slate-700'}>{c.supportCalls}</strong></div>
                            <div className="text-[10px] text-slate-400">Monthly spend: ${c.monthlyCharges}</div>
                          </td>
                          <td className="p-3">
                            <span className="capitalize">{c.contractType}</span>
                            <div className="text-[10px] text-slate-400">{c.billingIssues === 1 ? '⚠️ Billing dispute' : 'No dispute'}</div>
                          </td>
                          <td className="p-3">
                            {c.churned === 1 ? (
                              <span className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full bg-rose-50 text-rose-600 border border-rose-100 text-[10px] font-semibold">
                                <BadgeAlert className="h-2.5 w-2.5" /> Churned
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-100 text-[10px] font-semibold">
                                Loyal
                              </span>
                            )}
                          </td>
                          <td className="p-3 font-mono">
                            <div className="flex items-center gap-1.5">
                              <span className={`w-2 h-2 rounded-full ${
                                c.predictedProb >= decisionThreshold ? 'bg-rose-500' : 'bg-emerald-500'
                              }`} />
                              <span className="font-bold">{(c.predictedProb * 100).toFixed(0)}%</span>
                            </div>
                            <div className="text-[9px] text-slate-400">
                              {c.predictedLabel === 1 ? 'Pred Leave' : 'Pred Stay'}
                            </div>
                          </td>
                          <td className="p-3 text-right">
                            <ChevronRight className="h-4 w-4 text-slate-300 ml-auto" />
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          /* Custom customer sandbox inputs */
          <div className="space-y-4">
            <div className="bg-amber-50/45 p-4 rounded-xl border border-amber-100 text-xs text-amber-900 leading-relaxed">
              <span className="font-bold flex items-center gap-1.5 mb-1 text-amber-800">
                <Lightbulb className="h-4 w-4" />
                Customer Sandbox Active
              </span>
              Tweak parameters below to simulate unique high-touch situations (such as monthly contract clients triggering 6 support disputes). The active neural network model will dynamically predict their risk percentages.
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Slider variables */}
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-600 font-semibold">
                    <span>Tenure Length (Months)</span>
                    <span className="font-mono text-slate-900">{sandboxTenure}m</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="72"
                    value={sandboxTenure}
                    onChange={(e) => setSandboxTenure(Number(e.target.value))}
                    className="w-full accent-indigo-600 h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-600 font-semibold">
                    <span>Monthly Charges ($)</span>
                    <span className="font-mono text-slate-900">${sandboxCharges}</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="160"
                    value={sandboxCharges}
                    onChange={(e) => setSandboxCharges(Number(e.target.value))}
                    className="w-full accent-indigo-600 h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-600 font-semibold">
                    <span>Usage Pattern Factor <span className="font-normal text-[10px] text-slate-400">(Relative to Average)</span></span>
                    <span className="font-mono text-slate-900">{sandboxUsage}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={sandboxUsage}
                    onChange={(e) => setSandboxUsage(Number(e.target.value))}
                    className="w-full accent-indigo-600 h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>

              {/* Contract options and calls */}
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-slate-600 font-semibold">
                    <span>Support Calls Count</span>
                    <span className="font-mono text-slate-900">{sandboxSupport} calls</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    value={sandboxSupport}
                    onChange={(e) => setSandboxSupport(Number(e.target.value))}
                    className="w-full accent-indigo-600 h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="space-y-1">
                  <span className="text-xs text-slate-600 font-semibold">Contract Style</span>
                  <div className="grid grid-cols-3 gap-2 mt-1">
                    {['month-to-month', 'one-year', 'two-year'].map(c => (
                      <button
                        key={c}
                        onClick={() => setSandboxContract(c as any)}
                        className={`py-1.5 px-2 rounded-lg border text-xs capitalize cursor-pointer font-semibold transition-all ${
                          sandboxContract === c 
                            ? 'bg-indigo-600 border-indigo-700 text-white' 
                            : 'bg-white border-slate-200 text-slate-600 text-[10px] hover:bg-slate-50'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1">
                  <span className="text-xs text-slate-600 font-semibold">Billing Dispute Active?</span>
                  <div className="flex items-center gap-4 mt-2 font-semibold">
                    <button
                      onClick={() => setSandboxBilling(1)}
                      className={`text-xs px-4 py-1.5 rounded-lg border cursor-pointer transition-all ${
                        sandboxBilling === 1 
                          ? 'bg-rose-500 border-rose-600 text-white shadow-sm' 
                          : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      Active Dispute
                    </button>
                    <button
                      onClick={() => setSandboxBilling(0)}
                      className={`text-xs px-4 py-1.5 rounded-lg border cursor-pointer transition-all ${
                        sandboxBilling === 0 
                          ? 'bg-emerald-500 border-emerald-600 text-white shadow-sm' 
                          : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      Normal Billing
                    </button>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

      </div>

      {/* RHS: Active selected Customer details card / AI strategy generation (5 cols) */}
      <div className="xl:col-span-5 flex flex-col gap-4">
        
        {/* Core Prediction panel */}
        <div className="bg-white border border-slate-100 rounded-xl p-5 shadow-sm space-y-4 flex-1">
          <h3 className="text-sm font-bold text-slate-800 pb-3 border-b border-slate-50 flex items-center gap-1.5">
            <Brain className="h-4.5 w-4.5 text-indigo-500 animate-pulse" />
            Predictive Scorecard Deep Dive
          </h3>

          {selectedCustomerInfo ? (
            <div className="space-y-5">
              
              {/* Profile overview */}
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="text-base font-bold text-slate-800">{selectedCustomerInfo.name}</h4>
                  <p className="text-xs font-mono text-slate-400 uppercase tracking-widest">{selectedCustomerInfo.id}</p>
                </div>
                
                {/* Contract badge */}
                <span className="text-[10px] bg-slate-100 border border-slate-200 rounded px-2 py-0.5 font-bold text-slate-600 capitalize">
                  {selectedCustomerInfo.contractType} Contract
                </span>
              </div>

              {/* Risk Gauge Bar */}
              <div className="space-y-2 bg-slate-50/50 rounded-xl p-4 border border-slate-100">
                <div className="flex justify-between items-end">
                  <span className="text-xs font-semibold text-slate-600">Calculated Churn Probability:</span>
                  <div className="text-right">
                    <span className={`text-xl font-black font-mono px-2.5 py-1.5 rounded-lg border ${getRiskBadgeColor(selectedCustomerInfo.predictedProb)}`}>
                      {(selectedCustomerInfo.predictedProb * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden mt-3 shadow-none">
                  <div 
                    className={`h-full transition-all duration-700 rounded-full ${
                      selectedCustomerInfo.predictedProb < 0.3 ? 'bg-emerald-500' :
                      selectedCustomerInfo.predictedProb < 0.6 ? 'bg-amber-500' : 'bg-rose-500'
                    }`}
                    style={{ width: `${selectedCustomerInfo.predictedProb * 100}%` }}
                  />
                </div>

                <p className="text-[11px] text-slate-500 pt-1 leading-relaxed">
                  <strong>Risk Assessment:</strong> {getRiskExplanation(selectedCustomerInfo.predictedProb)}
                </p>
              </div>

              {/* Static factor check list */}
              <div className="grid grid-cols-2 gap-3 text-xs leading-normal">
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <span className="text-slate-400 font-semibold block text-[10px] uppercase">Account Age</span>
                  <strong className="text-slate-700">{selectedCustomerInfo.tenure} Months</strong>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <span className="text-slate-400 font-semibold block text-[10px] uppercase">Monthly Charges</span>
                  <strong className="text-slate-700">${selectedCustomerInfo.monthlyCharges} Spend</strong>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <span className="text-slate-400 font-semibold block text-[10px] uppercase">Usage Factor</span>
                  <strong className="text-slate-700">{selectedCustomerInfo.usageFactor}x Actives</strong>
                </div>
                <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <span className="text-slate-400 font-semibold block text-[10px] uppercase">Billing Issue</span>
                  <strong className={selectedCustomerInfo.billingIssues === 1 ? 'text-rose-600' : 'text-slate-700'}>
                    {selectedCustomerInfo.billingIssues === 1 ? 'Active Complications' : 'Healthy Record'}
                  </strong>
                </div>
              </div>

              {/* Propose AI outreach Strategy play */}
              <div className="pt-2 border-t border-slate-50">
                <button
                  onClick={handleGeneratePlaybook}
                  disabled={isGeneratingPlaybook}
                  className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow font-semibold text-xs flex items-center justify-center gap-1.5 cursor-pointer disabled:opacity-50 transition-all duration-150 active:scale-95"
                >
                  <Sparkles className="h-4 w-4 animate-bounce" />
                  {isGeneratingPlaybook ? 'Drafting Playbook via Gemini 3.5...' : 'Generate AI Retention Playbook'}
                </button>
                <span className="text-[10px] text-slate-400 block text-center mt-1.5 leading-normal">
                  Powered by <strong>Gemini 3.5 Flash</strong>. Custom accounts diagnostics and high-touch customer Success email pitches drafted immediately based on their actual parameters.
                </span>
              </div>

            </div>
          ) : (
            <div className="text-center p-8 text-slate-400">
              Please choose a customer from the left to display scorecard coordinates.
            </div>
          )}

        </div>

        {/* Playbook result area */}
        {(isGeneratingPlaybook || generatedPlaybook || playbookError) && (
          <div className="bg-slate-900 text-slate-100 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4 max-h-[400px] overflow-y-auto duration-200 shrink-0">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <Server className="h-4 w-4 text-emerald-400 animate-pulse" />
                <span className="text-xs font-bold text-slate-200">Playbook Engine Response</span>
              </div>
              <span className="text-[10px] bg-slate-800 text-slate-400 font-mono px-2 py-0.5 rounded border border-slate-700">
                Status: {isGeneratingPlaybook ? 'COMPUTING' : 'SUCCESS'}
              </span>
            </div>

            {isGeneratingPlaybook && (
              <div className="space-y-3.5 py-4">
                <div className="flex items-center gap-2 text-xs text-indigo-300 font-bold">
                  <div className="border-t-2 border-r-2 border-indigo-400 w-4 h-4 rounded-full animate-spin mr-1" />
                  <span>Analyzing account triggers...</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full w-3/4 animate-pulse" />
                <div className="h-2 bg-slate-800 rounded-full w-5/6 animate-pulse" />
                <div className="h-2 bg-slate-800 rounded-full w-1/2 animate-pulse" />
              </div>
            )}

            {playbookError && (
              <div className="p-3 bg-rose-950/40 border border-rose-900 rounded-lg text-xs text-rose-300">
                <span className="font-bold block mb-1">Playbook Error</span>
                {playbookError}
              </div>
            )}

            {generatedPlaybook && (
              <div className="text-xs leading-relaxed space-y-3 prose prose-invert max-w-none text-slate-300 overflow-wrap-normal">
                {/* Convert Markdown double asterisks or headers safely to display elements */}
                {generatedPlaybook.split('\n').map((line, idx) => {
                  if (line.trim().startsWith('###') || line.trim().startsWith('**')) {
                    return <p key={idx} className="font-bold text-indigo-400 mt-3 mb-1 text-sm">{line.replace(/[#*]/g, '')}</p>;
                  }
                  if (line.trim().startsWith('-') || line.trim().startsWith('*')) {
                    return <li key={idx} className="ml-4 list-disc text-slate-300 my-1">{line.replace(/^[-*]\s*/, '')}</li>;
                  }
                  return <p key={idx} className="my-1.5">{line}</p>;
                })}
              </div>
            )}
          </div>
        )}

      </div>

    </div>
  );
}
