// If using npm with a bundler, import numeric as follows:
const numeric = require('numeric');

/**************************************
 * UTILITY FUNCTIONS
 **************************************/

/**
 * Returns a normally distributed random number.
 * @param {number} mean 
 * @param {number} stdev 
 * @returns {number}
 */
function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random(); // (0,1]
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

/**
 * Sigmoid function.
 * @param {number} x 
 * @returns {number}
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Logit function.
 * @param {number} x - value between 0 and 1.
 * @returns {number} logit(x)
 */
function logit(x) {
  // Prevent extreme values
  if (x <= 0) x = 0.01;
  if (x >= 1) x = 0.99;
  return Math.log(x / (1 - x));
}

/**************************************
 * PARAMETER ESTIMATION FUNCTIONS
 **************************************/

/**
 * Approximates the mean and standard deviation of a LogitNormal distribution
 * with parameters (-d, s) using the delta method.
 *
 * For LogitNormal(-d, s):
 *   μ = -d,
 *   u = sigmoid(μ),
 *   f'(μ) = u * (1 - u),
 *   f''(μ) = u * (1 - u) * (1 - 2u)
 *
 * Then:
 *   muApprox  = u + 0.5 * f''(μ) * s^2
 *   stdApprox = f'(μ) * s
 *
 * @param {number} d - difficulty parameter.
 * @param {number} s - standard deviation parameter.
 * @returns {{ muApprox: number, stdApprox: number }}
 */
// function approxMoments(d, s) {
//   const mu = -d;
//   const u = sigmoid(mu);
//   const fPrime = u * (1 - u);
//   const fDoublePrime = u * (1 - u) * (1 - 2 * u);
//   const muApprox = u + 0.5 * fDoublePrime * s * s;
//   const stdApprox = fPrime * s;
//   return { muApprox, stdApprox };
// }

const tf = require('@tensorflow/tfjs'); 
// function approxMoments(d, s) {
//   // Step 1: Generate samples from a Normal distribution with mean -d and stdDev s
//   const normalSamples = tf.randomNormal([10000], -d, s, 'float32');

//   // Step 2: Transform the samples into a Logit-Normal distribution using the Sigmoid function
//   const logitNormalSamples = tf.sigmoid(normalSamples);

//   // Step 3: Approximate the mean and standard deviation of the Logit-Normal distribution
//   const muApprox = logitNormalSamples.mean().arraySync(); // Approximate mean
//   // Step 4: Calculate the standard deviation of the Logit-Normal distribution
//   const squaredDiffs = logitNormalSamples.sub(muApprox).square(); // (x_i - μ)^2
//   const variance = squaredDiffs.mean(); // Mean of squared differences
//   const stdApprox = variance.sqrt().arraySync(); // Square root of variance

//   console.log("Mu approx", typeof(muApprox), muApprox)
//   console.log("STD approx", typeof(stdApprox), stdApprox)

//   return [muApprox, stdApprox];
// }

// function approxMoments(d, s) {
//   // Step 1: Generate samples from a Normal distribution with mean -d and stdDev s
//   const normalSamples = tf.randomNormal([10000], -d, s, 'float32');

//   // Step 2: Transform the samples into a Logit-Normal distribution using the Sigmoid function
//   const logitNormalSamples = tf.sigmoid(normalSamples);

//   // Step 3: Approximate the mean and standard deviation of the Logit-Normal distribution
//   const muApprox = logitNormalSamples.mean().arraySync(); // Approximate mean

//   // Step 4: Calculate the standard deviation of the Logit-Normal distribution
//   const squaredDiffs = logitNormalSamples.sub(muApprox).square(); // (x_i - μ)^2
//   const variance = squaredDiffs.mean(); // Mean of squared differences
//   const stdApprox = variance.sqrt().arraySync(); // Square root of variance

//   // Ensure valid values
//   if (isNaN(muApprox) || isNaN(stdApprox)) {
//     throw new Error(`Invalid muApprox or stdApprox: d=${d}, s=${s}`);
//   }

//   return { muApprox, stdApprox };
// }

function approxMoments(d, s) {
  // d: TensorFlow tensor (shape [1])
  // s: TensorFlow tensor (shape [1]) representing the standard deviation
  // For LogitNormal(-d, s), we set:
  //   μ = -d,
  //   u = sigmoid(μ) => here, μ = -d so u = sigmoid(-d)
  //   f'(μ) = u * (1 - u)
  //   f''(μ) = u * (1 - u) * (1 - 2u)
  // Then:
  //   muApprox = u + 0.5 * f''(μ) * s^2
  //   stdApprox = f'(μ) * s

  const u = tf.sigmoid(tf.neg(d)); // u = sigmoid(-d)
  console.log("tf.scalar(1).dtype:", tf.scalar(1).dtype); // should output a numeric type like 'float32'
  console.log("d dtype:", d.dtype);
  console.log("s dtype:", s.dtype);
  const oneMinusU = tf.sub(tf.scalar(1), u); // 1 - u
  const fPrime = u.mul(oneMinusU); // f'(μ) = u * (1 - u)
  const fDoublePrime = fPrime.mul(tf.sub(tf.scalar(1), tf.mul(tf.scalar(2), u))); // f''(μ) = u*(1-u)*(1-2u)
  
  const muApprox = u.add(tf.mul(tf.scalar(0.5), fDoublePrime.mul(tf.square(s))));
  const stdApprox = fPrime.mul(s);
  
  return { muApprox, stdApprox };
}


/**
 * Objective function to minimize.
 * @param {Array<number>} params - [d, logVar]
 * @param {number} mu - observed mean.
 * @param {number} std - observed standard deviation.
 * @param {*} _ - dummy parameter.
 * @returns {number} loss value.
 */
// function objective(params, mu, std, _) {
//   const paramsArray = params.arraySync()
//   const [d, logVar] = paramsArray;
//   const variance = Math.exp(logVar);
//   const stdDev = Math.sqrt(variance);
  
//   const { muApprox, stdApprox } = approxMoments(d, stdDev);
//   const loss = Math.pow(mu - muApprox, 2) + Math.pow(std - stdApprox, 2);
//   // Return the loss as a TensorFlow tensor
//   return tf.scalar(loss);
// }

function simpleObjective(params) {
  // Ensure params is a TensorFlow tensor
  if (!(params instanceof tf.Tensor)) {
    throw new Error(`params is not a tensor: ${params}`);
  }

  // Compute f(x) = x^2
  const x = params.slice([0], [1]); // Extract the first element of the tensor
  const loss = tf.square(x).squeeze();  // Convert to a scalar

  // Log intermediate values for debugging
  console.log(`x: ${x.arraySync()[0]}, loss: ${loss.arraySync()}`);

  // Return the loss as a TensorFlow tensor (now scalar)
  return loss;
}


function objective(params, mu, std, _) {
  // Extract d and logVar as tensors.
  const d = params.slice([0], [1]);      // shape [1]
  const logVar = params.slice([1], [1]);   // shape [1]

  // Compute variance and standard deviation using TensorFlow operations.
  const variance = tf.exp(logVar);
  const stdDev = tf.sqrt(variance);

  // Call the TensorFlow-based approxMoments helper.
  const { muApprox, stdApprox } = approxMoments(d, stdDev);

  console.log("MU tye", tf.scalar(parseFloat(mu)).dtype)
  console.log("std type", tf.scalar(std).dtype)

  // Compute the loss as the sum of squared differences.
  const loss = tf.add(
    tf.square(tf.sub(tf.scalar(parseFloat(mu)), muApprox)),
    tf.square(tf.sub(tf.scalar(std), stdApprox))
  );

  // Optionally log intermediate values (outside gradient computation, or wrapped in tf.tidy).
  tf.tidy(() => {
    console.log("d:", d.arraySync()[0]);
    console.log("logVar:", logVar.arraySync()[0]);
    console.log("muApprox:", muApprox.arraySync()[0]);
    console.log("stdApprox:", stdApprox.arraySync()[0]);
    console.log("loss:", loss.arraySync()[0]);
  });

  // Return the loss as a scalar.
  return loss.squeeze();
}


// /**
//  * Minimizes a function using numeric.uncmin (BFGS-style optimization).
//  * @param {Function} objectiveFn - objective function.
//  * @param {Array<number>} initialGuess - initial parameters [d, logVar].
//  * @param {Array} args - additional arguments for objectiveFn.
//  * @returns {{ x: Array<number>, fun: number }} optimization result.
//  */
// function minimizeBFGS(objectiveFn, initialGuess, args = []) {
//   // Wrap objective function to accept a vector x.
//   const wrappedObjective = function(x) {
//     return objectiveFn(x, ...args);
//   };
//   const result = numeric.uncmin(wrappedObjective, initialGuess);
//   // result.solution holds optimized parameters; result.f is the final loss.
//   return { x: result.solution, fun: result.f };
// }

/**
 * Minimizes a function using TensorFlow.js gradient-based optimization.
 * @param {Function} objectiveFn - Objective function to minimize.
 * @param {Array<number>} initialGuess - Initial parameters [d, logVar].
 * @param {Array} args - Additional arguments for objectiveFn.
 * @param {Object} options - Optimization options (e.g., learning rate, max iterations).
 * @returns {{ x: Array<number>, fun: number }} Optimization result.
 */
async function minimizeBFGS(objectiveFn, initialGuess, args = [], options = {}) {
  const { learningRate = 0.01, maxIterations = 1000 } = options;

  // Convert initialGuess to a TensorFlow variable (trainable parameters)
  const params = tf.variable(tf.tensor1d(initialGuess));
  console.log("PARAMS", params);

  // Define the optimizer (e.g., Adam)
  const optimizer = tf.train.adam(learningRate);

  let bestParams = null;
  let bestLoss = Infinity;

  // Optimization loop
  for (let i = 0; i < maxIterations; i++) {
    try {
      optimizer.minimize(() => {
        // Directly use params in the loss function
        const loss = objectiveFn(params, ...args);
        return loss; // Ensure the loss is a tensor
      });

      // Track the best parameters
      const loss = objectiveFn(params, ...args);
      const currentLoss = loss.arraySync(); // Convert tensor to a number
      if (currentLoss < bestLoss) {
        bestLoss = currentLoss;
        bestParams = params.arraySync(); // Ensure bestParams is an array
      }

      // Print progress (optional)
      if (i % 100 === 0) {
        console.log(`Iteration ${i}: Loss = ${currentLoss}`);
      }
    } catch (error) {
      console.error(`Optimization error at iteration ${i}: ${error}`);
      break;
    }
  }

  // Ensure bestParams is an array
  if (!bestParams) throw new Error("Optimization failed: No valid parameters found");

  // Get the optimized parameters and final loss
  const optimizedParams = bestParams;
  const finalLoss = objectiveFn(params, ...args).arraySync();

  // Dispose of tensors to free memory
  params.dispose();

  return { x: optimizedParams, fun: finalLoss };
}

/**
 * Estimates parameters (difficulty and epsilon squared) by minimizing the objective.
 * @param {number} mu - observed mean.
 * @param {number} std - observed standard deviation.
 * @param {number} priorAbilityStd - prior ability standard deviation (unused here but kept for interface).
 * @returns {{ dOpt: number, epsilonSquaredOpt: number, distance: number }}
 */

async function estimateParameters(mu, std, priorAbilityStd) {
  const dOpt = -4;
  const epsilonSquaredOpt = 0.28;
  const distance = 12;
  return { dOpt, epsilonSquaredOpt, distance };
}


async function oldEstimateParameters(mu, std, priorAbilityStd) {
  let bestResult = null;
  let bestLoss = Infinity;

  // Try 3 random initializations.
  for (let i = 0; i < 3; i++) {
    const initialD = Math.random() * 20 - 10; // Uniform in [-10, 10]
    const initialLogVar = Math.log(Math.random() * (5 - 0.001) + 0.001); // log(random in [0.001, 5])
    const initialGuess = [initialD, initialLogVar];

    try {
      const result = await minimizeBFGS(objective, initialGuess, [mu, std, 0]);
      console.log("Result!!!!", result)
      if (result.fun < bestLoss) {
        bestLoss = result.fun;
        bestResult = result;
      }
    } catch (error) {
      console.error(`Optimization failed for initialization ${i}: ${error}`);
    }
  }

  if (!bestResult) throw new Error("Optimization failed for all initializations");

  const [dOpt, logVarOpt] = bestResult.x;
  const epsilonSquaredOpt = Math.exp(logVarOpt);
  const distance = bestResult.fun;
  return { dOpt, epsilonSquaredOpt, distance };
}

/**
 * Estimates class statistics based on observed assignment statistics.
 * @param {Object} observed - Object with an "assignment_stats" property.
 *        Example:
 *          {
 *            assignment_stats: {
 *              "assignment1": { mean: 0.7, std: 0.1 },
 *              "assignment2": { mean: 0.6, std: 0.12 },
 *              ...
 *            }
 *          }
 * @param {*} _ - dummy parameter.
 * @param {number} alpha - scaling factor for prior ability std, default is 1.
 * @returns {{
 *   classStats: { difficulties: number[], assn_epsilons: number[], S: number },
 *   totalDistance: number[]
 * }}
 */
async function estimateStats(observed, _, alpha = 0.0001) {
  const assnStats = observed.assignment_stats;
  const difficulties = [];
  const logitVars = [];
  const totalDistance = [];

  console.log("ASSN STATS", assnStats);

  // Use a for-loop to handle async/await properly
  for (const key of Object.keys(assnStats)) {
    const { mean: muGrade, std: stdGrade } = assnStats[key];
    try {
      const { dOpt, epsilonSquaredOpt, distance } = await estimateParameters(muGrade, stdGrade, _);
      difficulties.push(dOpt);
      logitVars.push(epsilonSquaredOpt);
      totalDistance.push(distance);
    } catch (error) {
      console.error(`Error estimating parameters for assignment ${key}: ${error}`);
    }
  }

  const courseMinVJ = Math.min(...logitVars);
  const priorAbilityStd = Math.min(
    alpha * courseMinVJ,
    Math.max(Math.sqrt(courseMinVJ) - 0.01, 0.01)
  );
  const epsilons = logitVars.map(v => Math.sqrt(v - Math.pow(priorAbilityStd, 2)));

  const classStats = {
    difficulties,
    assn_epsilons: epsilons,
    S: priorAbilityStd
  };

  return { classStats, totalDistance };
}

/**************************************
 * ABILITY INFERENCE FUNCTIONS
 **************************************/

/**
 * Computes the Gaussian posterior for ability given observations and variances.
 * @param {Array<number>} obs - list of observed ability samples.
 * @param {Array<number>} vars - list of variances for each observation.
 * @param {number} S - prior ability standard deviation.
 * @returns {{ muPost: number, sigmaPostSquared: number }}
 */
function gaussianPosterior(obs, vars, S) {
  const muPrior = 0;
  let weightedSum = 0;
  let sumInverseVariances = 0;

  console.log("obs", obs)

  for (let i = 0; i < obs.length; i++) {
    console.log("obs[i]", typeof(obs[i]), obs[i])
    console.log("var[i]", typeof(vars[i]), vars[i])
    weightedSum += obs[i] / vars[i];
    sumInverseVariances += 1 / vars[i];
  }

  const precisionPrior = 1 / (S * S);
  const denominator = precisionPrior + sumInverseVariances;
  console.log("demonimnaotr", denominator)
  console.log("precision prior", precisionPrior)
  console.log("muPrior", muPrior)
  console.log("weightedSum", weightedSum)
  const muPost = (muPrior * precisionPrior + weightedSum) / denominator;
  const sigmaPostSquared = 1 / denominator;

  return { muPost, sigmaPostSquared };
}

/**
 * Returns the difficulty and epsilon for a given assignment index.
 * If the provided value is an array, returns its average; otherwise returns the value itself.
 * @param {Array|number} difficulties 
 * @param {Array|number} epsilons 
 * @param {number} j - assignment index.
 * @returns {{ difficulty: number, epsilon: number }}
 */
function getDifficultyEpsilon(difficulties, epsilons, j) {
  const softDifficulty = difficulties[j];
  const softEpsilon = epsilons[j];
  const diff_mu = Array.isArray(softDifficulty)
    ? softDifficulty.reduce((a, b) => a + b, 0) / softDifficulty.length
    : softDifficulty;
  const epsilon_mu = Array.isArray(softEpsilon)
    ? softEpsilon.reduce((a, b) => a + b, 0) / softEpsilon.length
    : softEpsilon;
  return { difficulty: diff_mu, epsilon: epsilon_mu };
}

/**
 * Estimates ability for a single student given their scores.
 * @param {Array<number>} scores - Array of scores for assignments.
 * @param {Array<number>} difficulties - Array of difficulties per assignment.
 * @param {Array<number>} epsilons - Array of epsilon values per assignment.
 * @param {number} S - Prior ability standard deviation.
 * @param {string|number} studentId - Identifier for the student.
 * @returns {{ mu: number, std: number }} Estimated ability as posterior mean and standard deviation.
 */
function estimateAbility(scores, difficulties, epsilons, S, studentId) {
  const abilitySamples = [];
  const variances = [];

  for (let j = 0; j < scores.length; j++) {
    const grade = scores[j];
    console.log("grade", grade)
    const { difficulty, epsilon } = getDifficultyEpsilon(difficulties, epsilons, j);
    // sample = logit(grade) + difficulty
    const sample = logit(grade) + difficulty;
    abilitySamples.push(sample);
    variances.push(epsilon * epsilon);
  }

  console.log("ability samples", abilitySamples)

  const { muPost, sigmaPostSquared } = gaussianPosterior(abilitySamples, variances, S);
  const abilityStd = Math.sqrt(sigmaPostSquared);

  console.log("muPost", muPost)

  return { mu: muPost, std: abilityStd };
}

/**
 * Estimates abilities for all students.
 * @param {Object} observed - Object with a "students" property.
 *        Example:
 *          {
 *            students: {
 *              "student1": { scores: [0.7, 0.8, ...] },
 *              "student2": { scores: [0.6, 0.75, ...] },
 *              ...
 *            }
 *          }
 * @param {Object} classStats - Object with class-level statistics { difficulties, assn_epsilons, S }.
 * @returns {Object} Mapping from student id to estimated ability { mu, std }.
 */
function estimateStudentAbilities(observed, classStats) {
  const studentAbilities = {};
  const difficulties = classStats.difficulties;
  const epsilons = classStats.assn_epsilons;

  Object.keys(observed.students).forEach(studentId => {
    const scores = observed.students[studentId].scores;
    const ability = estimateAbility(scores, difficulties, epsilons, classStats.S, studentId);
    studentAbilities[studentId] = ability;
  });
  return studentAbilities;
}

/**************************************
 * SOFT GRADE ESTIMATION FUNCTIONS
 **************************************/

/**
 * Samples a score given a student's ability, assignment difficulty, and epsilon.
 * Uses a normal random sample and transforms it with the sigmoid.
 * @param {number} ability 
 * @param {number} difficulty 
 * @param {number} epsilon 
 * @returns {number} Score between 0 and 1.
 */
function sampleScore(ability, difficulty, epsilon) {
  const abilitySample = gaussianRandom(ability, epsilon);
  const diff = abilitySample - difficulty;
  return sigmoid(diff);
}

/**
 * For a single student, generates a distribution of final grades and predicted future scores.
 * This function simulates nParticles samples.
 *
 * @param {Array<number>} scoresSoFar - Scores from assignments already completed.
 * @param {{ mu: number, std: number }} abilityEstimate - Estimated student ability.
 * @param {Array<number>} difficulties - Array of difficulties for all assignments.
 * @param {Array<number>} epsilons - Array of epsilons for all assignments.
 * @returns {{ finalGrades: Array<number>, futureScores: Array<number> }}
 */
function estimateSingleStudentGradeSoftSquared(scoresSoFar, abilityEstimate, difficulties, epsilons) {
  const nParticles = 100;
  const finalGrades = [];
  const nFuture = difficulties.length - scoresSoFar.length;
  const futureScores = Array(nFuture).fill(0);
  const diffValues = Array(nFuture).fill(0);
  const epsValues = Array(nFuture).fill(0);

  for (let i = 0; i < nParticles; i++) {
    // Copy scoresSoFar (shallow copy is fine for numbers)
    const scores = scoresSoFar.slice();
    // Sample an ability from the estimated ability distribution.
    const ability = gaussianRandom(abilityEstimate.mu, abilityEstimate.std);
    for (let j = scoresSoFar.length; j < difficulties.length; j++) {
      const { difficulty, epsilon } = getDifficultyEpsilon(difficulties, epsilons, j);
      const index = j - scoresSoFar.length;
      diffValues[index] += difficulty;
      epsValues[index] += epsilon;
      const score = sampleScore(ability, difficulty, epsilon);
      futureScores[index] += score;
      scores.push(score);
    }
    // Compute the mean of scores as the final grade.
    const gradeMean = scores.reduce((a, b) => a + b, 0) / scores.length;
    finalGrades.push(gradeMean);
  }
  // Average the collected future scores over all particles.
  const avgFutureScores = futureScores.map(val => val / nParticles);
  return { finalGrades, futureScores: avgFutureScores };
}

/**
 * Estimates soft grade distributions for each student based on ability estimates.
 * Iterates over each student, simulating future assignment outcomes.
 *
 * @param {Object} abilityEstimates - Mapping from student id to estimated ability { mu, std }.
 * @param {Object} classStats - Class-level statistics { difficulties, assn_epsilons, S }.
 * @param {Object} observed - Observed data, with a "students" property.
 *        Example:
 *          {
 *            students: {
 *              "student1": { scores: [0.7, 0.8, ...] },
 *              ...
 *            }
 *          }
 * @param {number} numAssn - (Optional) Number of assignments (unused here).
 * @returns {{ studentGrades: Object, studentFutureScores: Object }}
 */
function estimateStudentGradesSoft(abilityEstimates, classStats, observed, numAssn) {
  const studentGrades = {};
  const studentFutureScores = {};
  const difficulties = classStats.difficulties;
  const epsilons = classStats.assn_epsilons;
  
  // Iterate over each student.
  Object.keys(abilityEstimates).forEach(studentId => {
    const scores = observed.students[studentId].scores;
    try {
      const { finalGrades, futureScores } = estimateSingleStudentGradeSoftSquared(
        scores,
        abilityEstimates[studentId],
        difficulties,
        epsilons
      );
      studentGrades[studentId] = finalGrades;
      studentFutureScores[studentId] = futureScores;
    } catch (error) {
      console.error(`Error processing student ${studentId}: ${error}`);
      console.error("Epsilons: ", epsilons);
      console.error("Class S (ability prior std): ", classStats.S);
      console.error("Student ability std: ", abilityEstimates[studentId].std);
    }
  });
  
  return { studentGrades, studentFutureScores };
}
/**
 * Computes the full soft belief pipeline:
 * 1. Estimates class statistics from assignment_stats.
 * 2. Infers student abilities.
 * 3. Computes soft grade distributions.
 *
 * @param {Object} observed - Should have:
 *    {
 *      assignment_stats: { 
 *         [assignmentId]: { mean: number, std: number } 
 *      },
 *      students: {
 *         [studentId]: { scores: number[] }
 *      }
 *    }
 * @returns {Object} An object containing classStats, abilityEstimates, studentGrades, studentFutureScores, and totalDistance.
 */

async function computeSoftBelief(observed) {
  console.log("YOOOOOOO");
  // Step 1: Estimate class statistics from assignment stats.
  const { classStats, totalDistance } = await estimateStats(observed, 0);

  console.log('stats estimated', classStats);

  // Step 2: Infer student abilities using the estimated class statistics.
  const abilityEstimates = estimateStudentAbilities(observed, classStats);

  console.log("abilities estimated", abilityEstimates);

  // Step 3: Estimate soft grade distributions.
  // Here we assume the number of assignments is the number of keys in assignment_stats.
  const numAssn = Object.keys(observed.assignment_stats).length;
  const { studentGrades, studentFutureScores } = estimateStudentGradesSoft(
    abilityEstimates,
    classStats,
    observed,
    numAssn
  );

  console.log("estimated grades", studentGrades);

  return {
    classStats,
    abilityEstimates,
    studentGrades,
    studentFutureScores,
    totalDistance
  };
}
  
  // Export the function for use in your React component.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports.computeSoftBelief = computeSoftBelief;
  }
  
