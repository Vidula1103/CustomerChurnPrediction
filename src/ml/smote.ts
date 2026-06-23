export interface SMOTEPoint {
  features: number[];
  label: number;
  isSynthetic: boolean;
  parentIds?: string[]; // IDs or indexes of original points that interpolated this
}

function calculateEuclideanDistance(v1: number[], v2: number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
}

/**
 * SMOTE algorithm to balance classes by generating synthetic minority instances
 * @param X Features matrix
 * @param Y Labels array (0 or 1)
 * @param k Number of nearest neighbors to consider (default: 5)
 */
export function applySMOTE(
  X: number[][],
  Y: number[],
  k: number = 5
): { smoteX: number[][]; smoteY: number[]; syntheticFlags: boolean[] } {
  // 1. Separate original majority and minority classes
  const minorityIndices: number[] = [];
  const majorityIndices: number[] = [];
  
  for (let i = 0; i < Y.length; i++) {
    if (Y[i] === 1) {
      minorityIndices.push(i);
    } else {
      majorityIndices.push(i);
    }
  }
  
  const originalMinorityCount = minorityIndices.length;
  const majorityCount = majorityIndices.length;
  
  // If already balanced or empty, return original
  if (originalMinorityCount === 0 || majorityCount === 0 || originalMinorityCount >= majorityCount) {
    return {
      smoteX: [...X],
      smoteY: [...Y],
      syntheticFlags: Array(X.length).fill(false)
    };
  }
  
  const smoteX = X.map(row => [...row]);
  const smoteY = [...Y];
  const syntheticFlags = Array(X.length).fill(false);
  
  // 2. We want to generate enough synthetic minority samples to match majorityCount
  const numSyntheticNeeded = majorityCount - originalMinorityCount;
  
  // Clamp k to actual size of minority sample minus 1
  const activeK = Math.max(1, Math.min(k, originalMinorityCount - 1));
  
  for (let s = 0; s < numSyntheticNeeded; s++) {
    // Select a random point from the minority class
    const randomIndex = minorityIndices[Math.floor(Math.random() * originalMinorityCount)];
    const targetVector = X[randomIndex];
    
    // Find neighbors in minority class
    const neighbors: { index: number; distance: number }[] = [];
    for (const minIdx of minorityIndices) {
      if (minIdx === randomIndex) continue;
      const distance = calculateEuclideanDistance(targetVector, X[minIdx]);
      neighbors.push({ index: minIdx, distance });
    }
    
    // Sort neighbors by distance
    neighbors.sort((a, b) => a.distance - b.distance);
    
    // Select one neighbor randomly from top k
    const selectKRange = Math.min(activeK, neighbors.length);
    if (selectKRange === 0) continue; // In case there's only 1 minority item
    
    const chosenNeighborInfo = neighbors[Math.floor(Math.random() * selectKRange)];
    const neighborVector = X[chosenNeighborInfo.index];
    
    // Interpolate points for each feature
    const syntheticVector: number[] = [];
    const randAlpha = Math.random(); // random ratio strictly between 0 and 1
    
    for (let f = 0; f < targetVector.length; f++) {
      // Linear interpolation
      const val = targetVector[f] + randAlpha * (neighborVector[f] - targetVector[f]);
      // Round binary features to 0 or 1
      const isBinaryFeature = f >= 4; // ContractMonthToMonth, ContractOneYear, BillingIssues
      if (isBinaryFeature) {
        syntheticVector.push(val > 0.5 ? 1 : 0);
      } else {
        syntheticVector.push(Number(val.toFixed(4)));
      }
    }
    
    // Add synthetic sample
    smoteX.push(syntheticVector);
    smoteY.push(1);
    syntheticFlags.push(true);
  }
  
  return { smoteX, smoteY, syntheticFlags };
}
