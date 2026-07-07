/// Advanced Optimization Algorithms in Rust
/// ==========================================
/// 
/// High-performance implementations of:
/// - Hidden Markov Model (HMM) fitting via Baum-Welch
/// - Viterbi decoding for state sequences
/// - MCMC Metropolis-Hastings sampling
/// - MLE optimization via gradient descent
/// - Mutual Information calculation

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
#[cfg(feature = "python")]
use rand::distributions::{Distribution, Uniform};
#[cfg(feature = "python")]
use rand_distr::Normal;
#[cfg(feature = "python")]
use rand::thread_rng;

/// HMM Parameters structure
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Clone)]
pub struct HMMParams {
    pub n_states: usize,
    pub transition_matrix: Vec<Vec<f64>>,
    pub emission_means: Vec<f64>,
    pub emission_stds: Vec<f64>,
    pub initial_probs: Vec<f64>,
}

impl HMMParams {
    pub fn new(n_states: usize) -> Self {
        HMMParams {
            n_states,
            transition_matrix: vec![vec![1.0 / n_states as f64; n_states]; n_states],
            emission_means: vec![0.0; n_states],
            emission_stds: vec![1.0; n_states],
            initial_probs: vec![1.0 / n_states as f64; n_states],
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl HMMParams {
    #[new]
    fn py_new(n_states: usize) -> Self {
        Self::new(n_states)
    }
    
    fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("n_states", self.n_states)?;
        dict.set_item("transition_matrix", self.transition_matrix.clone())?;
        dict.set_item("emission_means", self.emission_means.clone())?;
        dict.set_item("emission_stds", self.emission_stds.clone())?;
        dict.set_item("initial_probs", self.initial_probs.clone())?;
        Ok(dict.into())
    }
}

/// Fit HMM using Baum-Welch algorithm (EM)
#[cfg_attr(feature = "python", pyfunction)]
pub fn fit_hmm(
    observations: Vec<f64>,
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
) -> HMMParams {
    let n_obs = observations.len();
    
    // Initialize parameters
    let mut params = HMMParams::new(n_states);
    
    // Initialize emission parameters using quantiles
    let mut sorted_obs = observations.clone();
    sorted_obs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    for i in 0..n_states {
        let start_idx = (i * n_obs) / n_states;
        let end_idx = ((i + 1) * n_obs) / n_states;
        let segment = &sorted_obs[start_idx..end_idx];
        
        params.emission_means[i] = segment.iter().sum::<f64>() / segment.len() as f64;
        let var: f64 = segment.iter()
            .map(|x| (x - params.emission_means[i]).powi(2))
            .sum::<f64>() / segment.len() as f64;
        params.emission_stds[i] = var.sqrt().max(1e-6);
    }
    
    // EM iterations
    let mut prev_log_likelihood = f64::NEG_INFINITY;
    
    for iter in 0..n_iterations {
        // E-step: Forward-Backward
        let alpha = forward(&observations, &params);
        let beta = backward(&observations, &params);
        let gamma = compute_gamma(&alpha, &beta);
        let xi = compute_xi(&observations, &params, &alpha, &beta);
        
        // M-step: Update parameters
        update_parameters(&observations, &mut params, &gamma, &xi);
        
        // Check convergence
        let log_likelihood = compute_log_likelihood(&alpha);
        
        if (log_likelihood - prev_log_likelihood).abs() < tolerance {
            println!("HMM converged at iteration {}", iter);
            break;
        }
        
        prev_log_likelihood = log_likelihood;
    }
    
    params
}

/// Forward algorithm: Compute alpha[t][s] = P(obs[0:t+1], state[t]=s)
fn forward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut alpha = vec![vec![0.0; n_states]; n_obs];
    
    // t=0
    for s in 0..n_states {
        alpha[0][s] = params.initial_probs[s] * emission_prob(observations[0], params, s);
    }
    
    // Normalize to prevent underflow
    let sum0: f64 = alpha[0].iter().sum();
    if sum0 > 0.0 {
        for s in 0..n_states {
            alpha[0][s] /= sum0;
        }
    }
    
    // t=1..T-1
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut sum = 0.0;
            for prev_s in 0..n_states {
                sum += alpha[t-1][prev_s] * params.transition_matrix[prev_s][s];
            }
            alpha[t][s] = sum * emission_prob(observations[t], params, s);
        }
        
        // Normalize at each step
        let sum_t: f64 = alpha[t].iter().sum();
        if sum_t > 0.0 {
            for s in 0..n_states {
                alpha[t][s] /= sum_t;
            }
        }
    }
    
    alpha
}

/// Backward algorithm: Compute beta[t][s] = P(obs[t+1:T] | state[t]=s)
fn backward(observations: &[f64], params: &HMMParams) -> Vec<Vec<f64>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut beta = vec![vec![0.0; n_states]; n_obs];
    
    // t=T-1
    for s in 0..n_states {
        beta[n_obs-1][s] = 1.0;
    }
    
    // t=T-2..0
    for t in (0..n_obs-1).rev() {
        for s in 0..n_states {
            let mut sum = 0.0;
            for next_s in 0..n_states {
                sum += params.transition_matrix[s][next_s] *
                       emission_prob(observations[t+1], params, next_s) *
                       beta[t+1][next_s];
            }
            beta[t][s] = sum;
        }
        
        // Normalize to prevent underflow
        let sum_t: f64 = beta[t].iter().sum();
        if sum_t > 0.0 {
            for s in 0..n_states {
                beta[t][s] /= sum_t;
            }
        }
    }
    
    beta
}

/// Compute gamma[t][s] = P(state[t]=s | obs)
fn compute_gamma(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_obs = alpha.len();
    let n_states = alpha[0].len();
    let mut gamma = vec![vec![0.0; n_states]; n_obs];
    
    for t in 0..n_obs {
        let sum: f64 = (0..n_states).map(|s| alpha[t][s] * beta[t][s]).sum();
        
        for s in 0..n_states {
            gamma[t][s] = alpha[t][s] * beta[t][s] / sum.max(1e-10);
        }
    }
    
    gamma
}

/// Compute xi[t][i][j] = P(state[t]=i, state[t+1]=j | obs)
fn compute_xi(
    observations: &[f64],
    params: &HMMParams,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
) -> Vec<Vec<Vec<f64>>> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    let mut xi = vec![vec![vec![0.0; n_states]; n_states]; n_obs-1];
    
    for t in 0..n_obs-1 {
        let mut sum = 0.0;
        
        for i in 0..n_states {
            for j in 0..n_states {
                xi[t][i][j] = alpha[t][i] *
                              params.transition_matrix[i][j] *
                              emission_prob(observations[t+1], params, j) *
                              beta[t+1][j];
                sum += xi[t][i][j];
            }
        }
        
        // Normalize
        for i in 0..n_states {
            for j in 0..n_states {
                xi[t][i][j] /= sum.max(1e-10);
            }
        }
    }
    
    xi
}

/// Update HMM parameters (M-step)
fn update_parameters(
    observations: &[f64],
    params: &mut HMMParams,
    gamma: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
) {
    let n_obs = observations.len();
    let n_states = params.n_states;
    
    // Update transition matrix
    for i in 0..n_states {
        let denom: f64 = gamma[..n_obs-1].iter().map(|g| g[i]).sum();
        
        for j in 0..n_states {
            let numer: f64 = xi.iter().map(|x| x[i][j]).sum();
            params.transition_matrix[i][j] = numer / denom.max(1e-10);
        }
    }
    
    // Update emission parameters
    for s in 0..n_states {
        let weights: Vec<f64> = gamma.iter().map(|g| g[s]).collect();
        let sum_weights: f64 = weights.iter().sum();
        
        // Weighted mean
        let mean = observations.iter()
            .zip(weights.iter())
            .map(|(obs, w)| obs * w)
            .sum::<f64>() / sum_weights.max(1e-10);
        
        // Weighted variance
        let var = observations.iter()
            .zip(weights.iter())
            .map(|(obs, w)| w * (obs - mean).powi(2))
            .sum::<f64>() / sum_weights.max(1e-10);
        
        params.emission_means[s] = mean;
        params.emission_stds[s] = var.sqrt().max(1e-6);
    }
}

/// Emission probability: P(obs | state) using Gaussian
fn emission_prob(observation: f64, params: &HMMParams, state: usize) -> f64 {
    let mean = params.emission_means[state];
    let std = params.emission_stds[state];
    
    let z = (observation - mean) / std;
    let coef = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());
    
    coef * (-0.5 * z * z).exp()
}

/// Compute log-likelihood
fn compute_log_likelihood(alpha: &[Vec<f64>]) -> f64 {
    let last_alpha = &alpha[alpha.len() - 1];
    let sum: f64 = last_alpha.iter().sum();
    sum.ln()
}

/// Viterbi algorithm: Find most likely state sequence
#[cfg_attr(feature = "python", pyfunction)]
pub fn viterbi_decode(observations: Vec<f64>, params: HMMParams) -> Vec<usize> {
    let n_obs = observations.len();
    let n_states = params.n_states;
    
    let mut delta = vec![vec![f64::NEG_INFINITY; n_states]; n_obs];
    let mut psi = vec![vec![0usize; n_states]; n_obs];
    
    // Initialize
    for s in 0..n_states {
        delta[0][s] = params.initial_probs[s].ln() + 
                      emission_prob(observations[0], &params, s).ln();
    }
    
    // Recursion
    for t in 1..n_obs {
        for s in 0..n_states {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_state = 0;
            
            for prev_s in 0..n_states {
                let val = delta[t-1][prev_s] + params.transition_matrix[prev_s][s].ln();
                if val > max_val {
                    max_val = val;
                    max_state = prev_s;
                }
            }
            
            psi[t][s] = max_state;
            delta[t][s] = max_val + emission_prob(observations[t], &params, s).ln();
        }
    }
    
    // Backtrack
    let mut path = vec![0usize; n_obs];
    let mut max_val = f64::NEG_INFINITY;
    let mut max_state = 0;
    for s in 0..n_states {
        if delta[n_obs-1][s] > max_val {
            max_val = delta[n_obs-1][s];
            max_state = s;
        }
    }
    path[n_obs-1] = max_state;
    
    for t in (0..n_obs-1).rev() {
        path[t] = psi[t+1][path[t+1]];
    }
    
    path
}

/// MCMC Metropolis-Hastings sampler
#[cfg(feature = "python")]
#[pyfunction]
pub fn mcmc_sample(
    log_likelihood_fn: &Bound<'_, PyAny>,
    data: Vec<f64>,
    initial_params: Vec<f64>,
    param_bounds: Vec<(f64, f64)>,
    n_samples: usize,
    burn_in: usize,
    proposal_std: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let mut rng = thread_rng();
    let n_params = initial_params.len();
    let mut current_params = initial_params.clone();
    let mut samples = Vec::new();
    
    // Compute initial log-likelihood
    let mut current_ll = log_likelihood_fn
        .call1((current_params.clone(), data.clone()))?
        .extract::<f64>()?;
    
    for iter in 0..n_samples + burn_in {
        // Propose new parameters
        let mut proposed_params = current_params.clone();
        
        for i in 0..n_params {
            let normal = Normal::new(0.0, proposal_std).unwrap();
            let delta = normal.sample(&mut rng);
            proposed_params[i] = (proposed_params[i] + delta)
                .max(param_bounds[i].0)
                .min(param_bounds[i].1);
        }
        
        // Compute proposed log-likelihood
        let proposed_ll = log_likelihood_fn
            .call1((proposed_params.clone(), data.clone()))?
            .extract::<f64>()?;
        
        // Acceptance probability
        let log_alpha = proposed_ll - current_ll;
        let uniform = Uniform::new(0.0, 1.0);
        let u: f64 = uniform.sample(&mut rng);
        
        if log_alpha > u.ln() {
            // Accept
            current_params = proposed_params;
            current_ll = proposed_ll;
        }
        
        // Save samples after burn-in
        if iter >= burn_in {
            samples.push(current_params.clone());
        }
    }
    
    Ok(samples)
}

/// Compute mutual information between X and Y
#[cfg_attr(feature = "python", pyfunction)]
pub fn mutual_information(
    x: Vec<f64>,
    y: Vec<f64>,
    n_bins: usize,
) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "X and Y must have same length");
    
    // Discretize into bins
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    let x_binned: Vec<usize> = x.iter()
        .map(|v| ((v - x_min) / (x_max - x_min) * (n_bins as f64)).floor() as usize)
        .map(|b| b.min(n_bins - 1))
        .collect();
    
    let y_binned: Vec<usize> = y.iter()
        .map(|v| ((v - y_min) / (y_max - y_min) * (n_bins as f64)).floor() as usize)
        .map(|b| b.min(n_bins - 1))
        .collect();
    
    // Compute joint and marginal probabilities
    let mut joint_counts = vec![vec![0usize; n_bins]; n_bins];
    let mut x_counts = vec![0usize; n_bins];
    let mut y_counts = vec![0usize; n_bins];
    
    for i in 0..n {
        joint_counts[x_binned[i]][y_binned[i]] += 1;
        x_counts[x_binned[i]] += 1;
        y_counts[y_binned[i]] += 1;
    }
    
    // Compute MI
    let mut mi = 0.0;
    
    for i in 0..n_bins {
        let px = x_counts[i] as f64 / n as f64;
        if px == 0.0 {
            continue;
        }
        
        for j in 0..n_bins {
            let py = y_counts[j] as f64 / n as f64;
            let pxy = joint_counts[i][j] as f64 / n as f64;
            
            if pxy > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    
    mi
}

/// Differential Evolution optimizer
#[cfg(feature = "python")]
#[pyfunction]
pub fn differential_evolution(
    objective_fn: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    popsize: usize,
    maxiter: usize,
    f: f64,  // mutation factor
    cr: f64, // crossover probability
) -> PyResult<(Vec<f64>, f64)> {
    let mut rng = thread_rng();
    let n_params = bounds.len();
    let pop_size = popsize * n_params;
    
    // Initialize population
    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            bounds.iter()
                .map(|(low, high)| {
                    let uniform = Uniform::new(*low, *high);
                    uniform.sample(&mut rng)
                })
                .collect()
        })
        .collect();
    
    // Evaluate initial population
    let mut fitness: Vec<f64> = population
        .iter()
        .map(|ind| {
            objective_fn
                .call1((ind.clone(),))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
        .collect();
    
    // Find initial best
    let mut best_idx = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    // Evolution loop
    for _iter in 0..maxiter {
        for i in 0..pop_size {
            // Select three random individuals (different from i)
            let mut indices: Vec<usize> = (0..pop_size).filter(|&idx| idx != i).collect();
            let uniform_idx = Uniform::new(0, indices.len());
            
            let a_idx = indices[uniform_idx.sample(&mut rng)];
            indices.retain(|&x| x != a_idx);
            let b_idx = indices[uniform_idx.sample(&mut rng) % indices.len()];
            indices.retain(|&x| x != b_idx);
            let c_idx = indices[uniform_idx.sample(&mut rng) % indices.len()];
            
            // Mutation: v = a + f * (b - c)
            let mutant: Vec<f64> = (0..n_params)
                .map(|j| {
                    let v = population[a_idx][j] + f * (population[b_idx][j] - population[c_idx][j]);
                    v.max(bounds[j].0).min(bounds[j].1)
                })
                .collect();
            
            // Crossover
            let uniform_prob = Uniform::new(0.0, 1.0);
            let j_rand = uniform_idx.sample(&mut rng) % n_params;
            let mut trial = population[i].clone();
            
            for j in 0..n_params {
                if uniform_prob.sample(&mut rng) < cr || j == j_rand {
                    trial[j] = mutant[j];
                }
            }
            
            // Selection
            let trial_fitness = objective_fn
                .call1((trial.clone(),))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY);
            
            if trial_fitness < fitness[i] {
                population[i] = trial;
                fitness[i] = trial_fitness;
                
                if trial_fitness < fitness[best_idx] {
                    best_idx = i;
                }
            }
        }
    }
    
    Ok((population[best_idx].clone(), fitness[best_idx]))
}

/// Grid search optimizer
#[cfg(feature = "python")]
#[pyfunction]
pub fn grid_search(
    objective_fn: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    n_points: usize,
) -> PyResult<(Vec<f64>, f64)> {
    let n_params = bounds.len();
    
    // Generate grid points for each dimension
    let grids: Vec<Vec<f64>> = bounds
        .iter()
        .map(|(low, high)| {
            (0..n_points)
                .map(|i| low + (high - low) * i as f64 / (n_points - 1) as f64)
                .collect()
        })
        .collect();
    
    let mut best_params = vec![0.0; n_params];
    let mut best_score = f64::NEG_INFINITY;
    
    // Recursive grid traversal
    fn evaluate_grid(
        objective_fn: &Bound<'_, PyAny>,
        grids: &[Vec<f64>],
        current: &mut Vec<f64>,
        depth: usize,
        best_params: &mut Vec<f64>,
        best_score: &mut f64,
    ) {
        if depth == grids.len() {
            // Evaluate current point
            if let Ok(result) = objective_fn.call1((current.clone(),)) {
                if let Ok(score) = result.extract::<f64>() {
                    if score > *best_score {
                        *best_score = score;
                        *best_params = current.clone();
                    }
                }
            }
            return;
        }
        
        for &value in &grids[depth] {
            current.push(value);
            evaluate_grid(objective_fn, grids, current, depth + 1, best_params, best_score);
            current.pop();
        }
    }
    
    let mut current = Vec::new();
    evaluate_grid(
        objective_fn,
        &grids,
        &mut current,
        0,
        &mut best_params,
        &mut best_score,
    );
    
    Ok((best_params, best_score))
}

/// Shannon entropy calculation
#[cfg_attr(feature = "python", pyfunction)]
pub fn shannon_entropy(x: Vec<f64>, n_bins: usize) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    
    // Find min/max
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    if (x_max - x_min).abs() < 1e-10 {
        return 0.0;
    }
    
    // Bin the data
    let bin_counts: Vec<usize> = x
        .iter()
        .map(|&val| {
            let bin = ((val - x_min) / (x_max - x_min) * (n_bins as f64 - 1.0)) as usize;
            bin.min(n_bins - 1)
        })
        .fold(vec![0; n_bins], |mut counts, bin| {
            counts[bin] += 1;
            counts
        });
    
    // Compute entropy: H(X) = -sum(p * log(p))
    let entropy: f64 = bin_counts
        .iter()
        .filter_map(|&count| {
            if count > 0 {
                let p = count as f64 / n as f64;
                Some(-p * p.ln())
            } else {
                None
            }
        })
        .sum();
    
    entropy
}

/// Register optimization functions
#[cfg(feature = "python")]
pub fn register_optimization_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent_module.py();
    let optimization_module = PyModule::new_bound(py, "optimization")?;
    
    optimization_module.add_class::<HMMParams>()?;
    optimization_module.add_function(wrap_pyfunction!(fit_hmm, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(viterbi_decode, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(mcmc_sample, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(mutual_information, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(differential_evolution, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(grid_search, &optimization_module)?)?;
    optimization_module.add_function(wrap_pyfunction!(shannon_entropy, &optimization_module)?)?;
    
    parent_module.add_submodule(&optimization_module)?;
    
    Ok(())
}
