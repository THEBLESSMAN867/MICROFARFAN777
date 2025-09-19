"""
Semantic Cubes Ontological Coding System
Advanced ML-powered value chain analysis and performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Data structure for performance metrics across hierarchical levels"""
    portfolio_id: str
    program_id: str
    project_id: str
    activity_id: str
    actual_value: float
    target_value: float
    metric_type: str
    timestamp: float
    hierarchy_level: str


@dataclass
class LossFunctionConfig:
    """Configuration parameters for loss functions"""
    penalty_severity: float = 1.0
    threshold_value: float = 0.1
    alpha: float = 1.0  # Exponential decay parameter
    beta: float = 2.0   # Convex function parameter
    epsilon: float = 1e-6  # Numerical stability


class Performance_Indicators_Loss_Functions:
    """
    ML-powered value chain analysis using supervised learning models to identify
    bottlenecks and inefficiencies across portfolio-program-project-activity hierarchies.
    
    Implements four operational loss function types with Gaussian Process regression
    for lead time prediction and uncertainty quantification.
    """
    
    def __init__(self, config: Optional[LossFunctionConfig] = None):
        """
        Initialize the performance indicators and loss functions system.
        
        Args:
            config: Configuration parameters for loss functions
        """
        self.config = config or LossFunctionConfig()
        self.performance_data: List[PerformanceMetrics] = []
        self.gp_models: Dict[str, GaussianProcessRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.trained_models: Dict[str, bool] = {}
        
        # Initialize GP kernels for hyperparameter optimization
        self.kernel_options = [
            ConstantKernel() * RBF() + WhiteKernel(),
            ConstantKernel() * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(),
            ConstantKernel() * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(),
        ]
        
        # Performance tracking
        self.bottleneck_analysis: Dict[str, Dict] = {}
        self.efficiency_scores: Dict[str, float] = {}
        
    def add_performance_data(self, metrics: Union[PerformanceMetrics, List[PerformanceMetrics]]):
        """Add performance metrics to the system"""
        if isinstance(metrics, PerformanceMetrics):
            self.performance_data.append(metrics)
        else:
            self.performance_data.extend(metrics)
    
    def quadratic_loss(self, actual: float, target: float, 
                      penalty_severity: Optional[float] = None) -> float:
        """
        Quadratic loss function: L(x) = penalty_severity * (actual - target)^2
        
        Args:
            actual: Actual performance value
            target: Target performance value
            penalty_severity: Multiplier for penalty severity
            
        Returns:
            Calculated quadratic loss value
        """
        penalty = penalty_severity or self.config.penalty_severity
        deviation = actual - target
        return penalty * (deviation ** 2)
    
    def exponential_loss(self, actual: float, target: float,
                        penalty_severity: Optional[float] = None,
                        alpha: Optional[float] = None) -> float:
        """
        Exponential loss function: L(x) = penalty_severity * (exp(alpha * |actual - target|) - 1)
        
        Args:
            actual: Actual performance value
            target: Target performance value
            penalty_severity: Multiplier for penalty severity
            alpha: Exponential decay parameter
            
        Returns:
            Calculated exponential loss value
        """
        penalty = penalty_severity or self.config.penalty_severity
        alpha = alpha or self.config.alpha
        deviation = abs(actual - target)
        return penalty * (np.exp(alpha * deviation) - 1)
    
    def linear_loss(self, actual: float, target: float,
                   penalty_severity: Optional[float] = None,
                   threshold: Optional[float] = None) -> float:
        """
        Linear loss function with threshold: L(x) = penalty_severity * max(0, |actual - target| - threshold)
        
        Args:
            actual: Actual performance value
            target: Target performance value
            penalty_severity: Multiplier for penalty severity
            threshold: Threshold below which no penalty is applied
            
        Returns:
            Calculated linear loss value
        """
        penalty = penalty_severity or self.config.penalty_severity
        threshold = threshold or self.config.threshold_value
        deviation = abs(actual - target)
        return penalty * max(0, deviation - threshold)
    
    def convex_loss(self, actual: float, target: float,
                   penalty_severity: Optional[float] = None,
                   beta: Optional[float] = None) -> float:
        """
        Convex loss function: L(x) = penalty_severity * |actual - target|^beta
        
        Args:
            actual: Actual performance value
            target: Target performance value
            penalty_severity: Multiplier for penalty severity
            beta: Convex function parameter (beta > 1 for convex)
            
        Returns:
            Calculated convex loss value
        """
        penalty = penalty_severity or self.config.penalty_severity
        beta = beta or self.config.beta
        deviation = abs(actual - target)
        return penalty * (deviation ** beta)
    
    def calculate_performance_degradation(self, loss_type: str = 'quadratic',
                                        **kwargs) -> Dict[str, float]:
        """
        Calculate performance degradation across all metrics using specified loss function.
        
        Args:
            loss_type: Type of loss function ('quadratic', 'exponential', 'linear', 'convex')
            **kwargs: Additional parameters for loss functions
            
        Returns:
            Dictionary of degradation scores by hierarchy level
        """
        if not self.performance_data:
            return {}
        
        loss_functions = {
            'quadratic': self.quadratic_loss,
            'exponential': self.exponential_loss,
            'linear': self.linear_loss,
            'convex': self.convex_loss
        }
        
        if loss_type not in loss_functions:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss_func = loss_functions[loss_type]
        degradation_scores = {}
        
        # Group by hierarchy levels
        hierarchy_groups = {}
        for metric in self.performance_data:
            level = metric.hierarchy_level
            if level not in hierarchy_groups:
                hierarchy_groups[level] = []
            hierarchy_groups[level].append(metric)
        
        # Calculate degradation for each level
        for level, metrics in hierarchy_groups.items():
            total_loss = 0.0
            count = 0
            
            for metric in metrics:
                loss = loss_func(metric.actual_value, metric.target_value, **kwargs)
                total_loss += loss
                count += 1
            
            if count > 0:
                degradation_scores[level] = total_loss / count
        
        return degradation_scores
    
    def identify_bottlenecks(self, threshold_percentile: float = 0.8) -> Dict[str, List[Dict]]:
        """
        Identify bottlenecks by analyzing performance degradation patterns.
        
        Args:
            threshold_percentile: Percentile threshold for identifying bottlenecks
            
        Returns:
            Dictionary of bottlenecks by hierarchy level
        """
        bottlenecks = {}
        
        # Calculate losses for all metrics using multiple loss functions
        loss_types = ['quadratic', 'exponential', 'linear', 'convex']
        
        for loss_type in loss_types:
            degradation_scores = self.calculate_performance_degradation(loss_type)
            
            for level, score in degradation_scores.items():
                if level not in bottlenecks:
                    bottlenecks[level] = []
                
                # Calculate threshold for this level
                level_metrics = [m for m in self.performance_data if m.hierarchy_level == level]
                level_losses = [self.quadratic_loss(m.actual_value, m.target_value) 
                              for m in level_metrics]
                
                if level_losses:
                    threshold = np.percentile(level_losses, threshold_percentile * 100)
                    
                    # Identify items above threshold
                    for metric in level_metrics:
                        loss = self.quadratic_loss(metric.actual_value, metric.target_value)
                        if loss > threshold:
                            bottleneck_info = {
                                'id': f"{metric.portfolio_id}/{metric.program_id}/{metric.project_id}/{metric.activity_id}",
                                'metric_type': metric.metric_type,
                                'actual_value': metric.actual_value,
                                'target_value': metric.target_value,
                                'loss': loss,
                                'loss_type': loss_type,
                                'deviation_percentage': abs(metric.actual_value - metric.target_value) / metric.target_value * 100
                            }
                            bottlenecks[level].append(bottleneck_info)
        
        self.bottleneck_analysis = bottlenecks
        return bottlenecks
    
    def prepare_gp_training_data(self, hierarchy_level: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for Gaussian Process regression.
        
        Args:
            hierarchy_level: Hierarchy level to prepare data for
            
        Returns:
            Tuple of features and target values
        """
        level_data = [m for m in self.performance_data if m.hierarchy_level == hierarchy_level]
        
        if not level_data:
            return np.array([]), np.array([])
        
        # Create feature matrix
        features = []
        targets = []
        
        for metric in level_data:
            feature_vector = [
                metric.actual_value,
                metric.target_value,
                abs(metric.actual_value - metric.target_value),  # deviation
                metric.timestamp,
                hash(metric.metric_type) % 1000,  # metric type encoding
            ]
            features.append(feature_vector)
            targets.append(metric.actual_value)  # Predict actual performance
        
        return np.array(features), np.array(targets)
    
    def optimize_gp_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        Optimize GP hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Optimized GaussianProcessRegressor
        """
        param_grid = []
        
        for kernel in self.kernel_options:
            param_grid.append({'kernel': [kernel]})
        
        gp = GaussianProcessRegressor(random_state=42, normalize_y=True)
        
        grid_search = GridSearchCV(
            gp, param_grid, cv=min(5, len(X)), 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def train_gp_models(self):
        """Train Gaussian Process models for each hierarchy level"""
        hierarchy_levels = set(m.hierarchy_level for m in self.performance_data)
        
        for level in hierarchy_levels:
            X, y = self.prepare_gp_training_data(level)
            
            if len(X) < 3:  # Need minimum samples for GP
                continue
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Optimize and train GP model
            gp_model = self.optimize_gp_hyperparameters(X_scaled, y)
            
            # Store model and scaler
            self.gp_models[level] = gp_model
            self.scalers[level] = scaler
            self.trained_models[level] = True
    
    def predict_lead_times(self, hierarchy_level: str, 
                          feature_vectors: np.ndarray,
                          return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict lead times using trained GP models with uncertainty quantification.
        
        Args:
            hierarchy_level: Hierarchy level for prediction
            feature_vectors: Input features for prediction
            return_std: Whether to return prediction uncertainty
            
        Returns:
            Predictions and optionally standard deviations
        """
        if hierarchy_level not in self.trained_models:
            raise ValueError(f"No trained model for hierarchy level: {hierarchy_level}")
        
        gp_model = self.gp_models[hierarchy_level]
        scaler = self.scalers[hierarchy_level]
        
        # Scale features
        X_scaled = scaler.transform(feature_vectors)
        
        # Make predictions
        if return_std:
            y_pred, y_std = gp_model.predict(X_scaled, return_std=True)
            return y_pred, y_std
        else:
            y_pred = gp_model.predict(X_scaled)
            return y_pred
    
    def calculate_prediction_intervals(self, predictions: np.ndarray, 
                                     uncertainties: np.ndarray,
                                     confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for uncertainty quantification.
        
        Args:
            predictions: Mean predictions
            uncertainties: Standard deviations
            confidence_level: Confidence level for intervals
            
        Returns:
            Lower and upper bounds of prediction intervals
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        return lower_bounds, upper_bounds
    
    def analyze_value_chain_efficiency(self) -> Dict[str, float]:
        """
        Analyze overall value chain efficiency using ML models.
        
        Returns:
            Efficiency scores by hierarchy level
        """
        efficiency_scores = {}
        
        # Train models if not already trained
        if not self.trained_models:
            self.train_gp_models()
        
        # Calculate efficiency for each level
        hierarchy_levels = set(m.hierarchy_level for m in self.performance_data)
        
        for level in hierarchy_levels:
            level_metrics = [m for m in self.performance_data if m.hierarchy_level == level]
            
            if not level_metrics:
                continue
            
            # Calculate efficiency as ratio of actual to target performance
            efficiency_ratios = []
            for metric in level_metrics:
                if metric.target_value != 0:
                    ratio = metric.actual_value / metric.target_value
                    efficiency_ratios.append(min(ratio, 2.0))  # Cap at 200%
            
            if efficiency_ratios:
                efficiency_scores[level] = np.mean(efficiency_ratios)
        
        self.efficiency_scores = efficiency_scores
        return efficiency_scores
    
    def generate_performance_report(self) -> Dict[str, any]:
        """
        Generate comprehensive performance analysis report.
        
        Returns:
            Complete performance analysis report
        """
        # Ensure models are trained
        if not self.trained_models:
            self.train_gp_models()
        
        # Analyze bottlenecks
        bottlenecks = self.identify_bottlenecks()
        
        # Calculate efficiency scores
        efficiency_scores = self.analyze_value_chain_efficiency()
        
        # Calculate degradation scores for all loss functions
        degradation_analysis = {}
        for loss_type in ['quadratic', 'exponential', 'linear', 'convex']:
            degradation_analysis[loss_type] = self.calculate_performance_degradation(loss_type)
        
        # Model performance metrics
        model_performance = {}
        for level, model in self.gp_models.items():
            if hasattr(model, 'log_marginal_likelihood_value_'):
                model_performance[level] = {
                    'log_marginal_likelihood': model.log_marginal_likelihood_value_,
                    'kernel': str(model.kernel_),
                    'n_samples': len([m for m in self.performance_data if m.hierarchy_level == level])
                }
        
        report = {
            'summary': {
                'total_metrics': len(self.performance_data),
                'hierarchy_levels': list(set(m.hierarchy_level for m in self.performance_data)),
                'trained_models': len(self.trained_models),
            },
            'efficiency_scores': efficiency_scores,
            'bottleneck_analysis': bottlenecks,
            'degradation_analysis': degradation_analysis,
            'model_performance': model_performance,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    config = LossFunctionConfig(penalty_severity=1.5, threshold_value=0.05)
    performance_system = Performance_Indicators_Loss_Functions(config)
    
    # Sample data generation
    import random
    sample_data = []
    
    for i in range(100):
        metric = PerformanceMetrics(
            portfolio_id=f"P{random.randint(1,3)}",
            program_id=f"PR{random.randint(1,5)}",
            project_id=f"PJ{random.randint(1,10)}",
            activity_id=f"A{random.randint(1,20)}",
            actual_value=random.uniform(0.5, 1.5),
            target_value=1.0,
            metric_type=random.choice(['duration', 'cost', 'quality', 'scope']),
            timestamp=random.uniform(1000, 2000),
            hierarchy_level=random.choice(['portfolio', 'program', 'project', 'activity'])
        )
        sample_data.append(metric)
    
    # Add sample data
    performance_system.add_performance_data(sample_data)
    
    # Train models
    performance_system.train_gp_models()
    
    # Generate report
    report = performance_system.generate_performance_report()
    
    print("Performance Analysis Report Generated Successfully!")
    print(f"Total metrics analyzed: {report['summary']['total_metrics']}")
    print(f"Hierarchy levels: {report['summary']['hierarchy_levels']}")
    print(f"Efficiency scores: {report['efficiency_scores']}")