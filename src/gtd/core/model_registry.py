"""Model registry: supported models with hyperparameter spaces and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HyperparameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    param_type: str  # "int", "float", "categorical", "bool"
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None
    log_scale: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "type": self.param_type,
            "default": self.default,
            "description": self.description,
        }
        if self.low is not None:
            result["low"] = self.low
        if self.high is not None:
            result["high"] = self.high
        if self.choices is not None:
            result["choices"] = self.choices
        if self.log_scale:
            result["log_scale"] = True
        return result


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a supported model."""

    name: str
    display_name: str
    description: str
    # e.g. ("binary_classification", "multiclass_classification", "regression")
    task_types: tuple[str, ...]
    sklearn_class: str  # Full import path
    hyperparameters: tuple[HyperparameterSpec, ...]
    supports_feature_importance: bool = True
    supports_predict_proba: bool = True
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "task_types": list(self.task_types),
            "hyperparameters": [hp.to_dict() for hp in self.hyperparameters],
            "supports_feature_importance": self.supports_feature_importance,
            "supports_predict_proba": self.supports_predict_proba,
            "tags": list(self.tags),
        }

    def get_default_params(self) -> dict[str, Any]:
        return {hp.name: hp.default for hp in self.hyperparameters}


# ─── Classification & Regression Models ───────────────────────────────────────

XGBOOST_SPEC = ModelSpec(
    name="xgboost",
    display_name="XGBoost",
    description="Gradient boosting; strong on structured/tabular data with good regularization",
    task_types=("binary_classification", "multiclass_classification", "regression"),
    sklearn_class="xgboost.XGBClassifier",  # or XGBRegressor
    hyperparameters=(
        HyperparameterSpec("n_estimators", "int", 300, 50, 2000),
        HyperparameterSpec("max_depth", "int", 6, 2, 15),
        HyperparameterSpec("learning_rate", "float", 0.1, 0.001, 1.0, log_scale=True),
        HyperparameterSpec("subsample", "float", 0.8, 0.5, 1.0),
        HyperparameterSpec("colsample_bytree", "float", 0.8, 0.3, 1.0),
        HyperparameterSpec("min_child_weight", "int", 1, 1, 20),
        HyperparameterSpec("gamma", "float", 0.0, 0.0, 5.0),
        HyperparameterSpec("reg_alpha", "float", 0.0, 0.0, 10.0),
        HyperparameterSpec("reg_lambda", "float", 1.0, 0.0, 10.0),
    ),
    tags=("gradient_boosting", "tree_based", "handles_missing"),
)

LIGHTGBM_SPEC = ModelSpec(
    name="lightgbm",
    display_name="LightGBM",
    description="Fast gradient boosting; handles categoricals natively, memory efficient",
    task_types=("binary_classification", "multiclass_classification", "regression"),
    sklearn_class="lightgbm.LGBMClassifier",
    hyperparameters=(
        HyperparameterSpec("n_estimators", "int", 300, 50, 2000),
        HyperparameterSpec("max_depth", "int", -1, -1, 15),
        HyperparameterSpec("learning_rate", "float", 0.1, 0.001, 1.0, log_scale=True),
        HyperparameterSpec("num_leaves", "int", 31, 8, 256),
        HyperparameterSpec("subsample", "float", 0.8, 0.5, 1.0),
        HyperparameterSpec("colsample_bytree", "float", 0.8, 0.3, 1.0),
        HyperparameterSpec("min_child_samples", "int", 20, 5, 100),
        HyperparameterSpec("reg_alpha", "float", 0.0, 0.0, 10.0),
        HyperparameterSpec("reg_lambda", "float", 0.0, 0.0, 10.0),
    ),
    tags=("gradient_boosting", "tree_based", "handles_categoricals", "fast_training"),
)

CATBOOST_SPEC = ModelSpec(
    name="catboost",
    display_name="CatBoost",
    description="Gradient boosting optimized for categorical features; robust to overfitting",
    task_types=("binary_classification", "multiclass_classification", "regression"),
    sklearn_class="catboost.CatBoostClassifier",
    hyperparameters=(
        HyperparameterSpec("iterations", "int", 500, 50, 2000),
        HyperparameterSpec("depth", "int", 6, 2, 12),
        HyperparameterSpec("learning_rate", "float", 0.1, 0.001, 1.0, log_scale=True),
        HyperparameterSpec("l2_leaf_reg", "float", 3.0, 0.1, 30.0, log_scale=True),
        HyperparameterSpec("subsample", "float", 0.8, 0.5, 1.0),
        HyperparameterSpec("colsample_bylevel", "float", 0.8, 0.3, 1.0),
        HyperparameterSpec("min_data_in_leaf", "int", 1, 1, 100),
        HyperparameterSpec("random_strength", "float", 1.0, 0.0, 10.0),
    ),
    tags=("gradient_boosting", "tree_based", "handles_categoricals", "robust"),
)

RANDOM_FOREST_SPEC = ModelSpec(
    name="random_forest",
    display_name="Random Forest",
    description="Ensemble of decision trees; robust baseline, handles noise well",
    task_types=("binary_classification", "multiclass_classification", "regression"),
    sklearn_class="sklearn.ensemble.RandomForestClassifier",
    hyperparameters=(
        HyperparameterSpec("n_estimators", "int", 200, 50, 1000),
        HyperparameterSpec("max_depth", "int", 10, 2, 30),
        HyperparameterSpec("min_samples_split", "int", 2, 2, 20),
        HyperparameterSpec("min_samples_leaf", "int", 1, 1, 20),
        HyperparameterSpec(
            "max_features", "categorical", "sqrt",
            choices=["sqrt", "log2", 0.5, 0.8, None],
        ),
        HyperparameterSpec("bootstrap", "bool", True),
        HyperparameterSpec("class_weight", "categorical", None, choices=["balanced", None]),
    ),
    tags=("ensemble", "tree_based", "robust"),
)

EXTRA_TREES_SPEC = ModelSpec(
    name="extra_trees",
    display_name="Extra Trees",
    description="Extremely randomized trees; faster than RF, good for high-dimensional data",
    task_types=("binary_classification", "multiclass_classification", "regression"),
    sklearn_class="sklearn.ensemble.ExtraTreesClassifier",
    hyperparameters=(
        HyperparameterSpec("n_estimators", "int", 200, 50, 1000),
        HyperparameterSpec("max_depth", "int", 10, 2, 30),
        HyperparameterSpec("min_samples_split", "int", 2, 2, 20),
        HyperparameterSpec("min_samples_leaf", "int", 1, 1, 20),
        HyperparameterSpec(
            "max_features", "categorical", "sqrt",
            choices=["sqrt", "log2", 0.5, 0.8, None],
        ),
        HyperparameterSpec("class_weight", "categorical", None, choices=["balanced", None]),
    ),
    tags=("ensemble", "tree_based", "fast_training"),
)

LOGISTIC_REGRESSION_SPEC = ModelSpec(
    name="logistic_regression",
    display_name="Logistic Regression",
    description="Interpretable linear classifier; good baseline for linearly separable problems",
    task_types=("binary_classification", "multiclass_classification"),
    sklearn_class="sklearn.linear_model.LogisticRegression",
    hyperparameters=(
        HyperparameterSpec("C", "float", 1.0, 0.001, 100.0, log_scale=True),
        HyperparameterSpec(
            "penalty", "categorical", "l2",
            choices=["l1", "l2", "elasticnet", None],
        ),
        HyperparameterSpec(
            "solver", "categorical", "lbfgs",
            choices=["lbfgs", "liblinear", "saga"],
        ),
        HyperparameterSpec("max_iter", "int", 1000, 100, 5000),
        HyperparameterSpec("class_weight", "categorical", None, choices=["balanced", None]),
    ),
    supports_feature_importance=False,
    tags=("linear", "interpretable", "fast_training"),
)

SVC_SPEC = ModelSpec(
    name="svc",
    display_name="Support Vector Classifier",
    description="Effective for small-medium datasets; finds optimal decision boundary",
    task_types=("binary_classification", "multiclass_classification"),
    sklearn_class="sklearn.svm.SVC",
    hyperparameters=(
        HyperparameterSpec("C", "float", 1.0, 0.01, 100.0, log_scale=True),
        HyperparameterSpec(
            "kernel", "categorical", "rbf",
            choices=["rbf", "linear", "poly", "sigmoid"],
        ),
        HyperparameterSpec("gamma", "categorical", "scale", choices=["scale", "auto"]),
        HyperparameterSpec("class_weight", "categorical", None, choices=["balanced", None]),
    ),
    supports_feature_importance=False,
    tags=("kernel", "small_data"),
)

KNN_CLASSIFIER_SPEC = ModelSpec(
    name="knn_classifier",
    display_name="K-Nearest Neighbors",
    description="Instance-based learning; good for low-dimensional data with clear clusters",
    task_types=("binary_classification", "multiclass_classification"),
    sklearn_class="sklearn.neighbors.KNeighborsClassifier",
    hyperparameters=(
        HyperparameterSpec("n_neighbors", "int", 5, 1, 50),
        HyperparameterSpec("weights", "categorical", "uniform", choices=["uniform", "distance"]),
        HyperparameterSpec(
            "metric", "categorical", "minkowski",
            choices=["minkowski", "euclidean", "manhattan"],
        ),
        HyperparameterSpec("p", "int", 2, 1, 3),
    ),
    supports_feature_importance=False,
    tags=("instance_based", "simple", "low_dimensional"),
)

MLP_CLASSIFIER_SPEC = ModelSpec(
    name="mlp_classifier",
    display_name="MLP Neural Network",
    description="Multi-layer perceptron; captures complex nonlinear patterns",
    task_types=("binary_classification", "multiclass_classification"),
    sklearn_class="sklearn.neural_network.MLPClassifier",
    hyperparameters=(
        HyperparameterSpec(
            "hidden_layer_sizes", "categorical", (100,),
            choices=[(64,), (128,), (100, 50), (128, 64), (256, 128, 64)],
        ),
        HyperparameterSpec("activation", "categorical", "relu", choices=["relu", "tanh"]),
        HyperparameterSpec("alpha", "float", 0.0001, 0.00001, 0.1, log_scale=True),
        HyperparameterSpec("learning_rate_init", "float", 0.001, 0.0001, 0.1, log_scale=True),
        HyperparameterSpec("max_iter", "int", 500, 100, 2000),
        HyperparameterSpec("batch_size", "categorical", "auto", choices=["auto", 32, 64, 128, 256]),
    ),
    supports_feature_importance=False,
    tags=("neural_network", "nonlinear"),
)

# ─── Regression-only models ───────────────────────────────────────────────────

LINEAR_REGRESSION_SPEC = ModelSpec(
    name="linear_regression",
    display_name="Linear Regression",
    description="Simple linear baseline; fast, interpretable",
    task_types=("regression",),
    sklearn_class="sklearn.linear_model.LinearRegression",
    hyperparameters=(),
    supports_feature_importance=False,
    supports_predict_proba=False,
    tags=("linear", "interpretable", "fast_training"),
)

ELASTICNET_SPEC = ModelSpec(
    name="elasticnet",
    display_name="ElasticNet",
    description="Linear regression with L1+L2 regularization; handles multicollinearity",
    task_types=("regression",),
    sklearn_class="sklearn.linear_model.ElasticNet",
    hyperparameters=(
        HyperparameterSpec("alpha", "float", 1.0, 0.001, 100.0, log_scale=True),
        HyperparameterSpec("l1_ratio", "float", 0.5, 0.0, 1.0),
        HyperparameterSpec("max_iter", "int", 1000, 100, 5000),
    ),
    supports_feature_importance=False,
    supports_predict_proba=False,
    tags=("linear", "regularized", "interpretable"),
)

SVR_SPEC = ModelSpec(
    name="svr",
    display_name="Support Vector Regressor",
    description="SVR for small-medium regression problems",
    task_types=("regression",),
    sklearn_class="sklearn.svm.SVR",
    hyperparameters=(
        HyperparameterSpec("C", "float", 1.0, 0.01, 100.0, log_scale=True),
        HyperparameterSpec("kernel", "categorical", "rbf", choices=["rbf", "linear", "poly"]),
        HyperparameterSpec("gamma", "categorical", "scale", choices=["scale", "auto"]),
        HyperparameterSpec("epsilon", "float", 0.1, 0.001, 1.0, log_scale=True),
    ),
    supports_feature_importance=False,
    supports_predict_proba=False,
    tags=("kernel", "small_data"),
)

KNN_REGRESSOR_SPEC = ModelSpec(
    name="knn_regressor",
    display_name="KNN Regressor",
    description="Instance-based regression; good for low-dimensional data",
    task_types=("regression",),
    sklearn_class="sklearn.neighbors.KNeighborsRegressor",
    hyperparameters=(
        HyperparameterSpec("n_neighbors", "int", 5, 1, 50),
        HyperparameterSpec("weights", "categorical", "uniform", choices=["uniform", "distance"]),
        HyperparameterSpec(
            "metric", "categorical", "minkowski",
            choices=["minkowski", "euclidean", "manhattan"],
        ),
    ),
    supports_feature_importance=False,
    supports_predict_proba=False,
    tags=("instance_based", "simple"),
)

MLP_REGRESSOR_SPEC = ModelSpec(
    name="mlp_regressor",
    display_name="MLP Regressor",
    description="Multi-layer perceptron for regression; captures nonlinear relationships",
    task_types=("regression",),
    sklearn_class="sklearn.neural_network.MLPRegressor",
    hyperparameters=(
        HyperparameterSpec(
            "hidden_layer_sizes", "categorical", (100,),
            choices=[(64,), (128,), (100, 50), (128, 64), (256, 128, 64)],
        ),
        HyperparameterSpec("activation", "categorical", "relu", choices=["relu", "tanh"]),
        HyperparameterSpec("alpha", "float", 0.0001, 0.00001, 0.1, log_scale=True),
        HyperparameterSpec("learning_rate_init", "float", 0.001, 0.0001, 0.1, log_scale=True),
        HyperparameterSpec("max_iter", "int", 500, 100, 2000),
    ),
    supports_feature_importance=False,
    supports_predict_proba=False,
    tags=("neural_network", "nonlinear"),
)

# ─── Pre-trained Models ───────────────────────────────────────────────────────

TABPFN_SPEC = ModelSpec(
    name="tabpfn",
    display_name="TabPFN",
    description=(
        "Pre-trained transformer for small tabular data (<10k rows, <100 features)"
    ),
    task_types=("binary_classification", "multiclass_classification"),
    sklearn_class="tabpfn.TabPFNClassifier",
    hyperparameters=(
        HyperparameterSpec("N_ensemble_configurations", "int", 32, 4, 64),
    ),
    supports_feature_importance=False,
    tags=("transformer", "small_data", "pretrained"),
)

# ─── Registry ─────────────────────────────────────────────────────────────────

ALL_MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        XGBOOST_SPEC,
        LIGHTGBM_SPEC,
        CATBOOST_SPEC,
        RANDOM_FOREST_SPEC,
        EXTRA_TREES_SPEC,
        LOGISTIC_REGRESSION_SPEC,
        SVC_SPEC,
        KNN_CLASSIFIER_SPEC,
        MLP_CLASSIFIER_SPEC,
        LINEAR_REGRESSION_SPEC,
        ELASTICNET_SPEC,
        SVR_SPEC,
        KNN_REGRESSOR_SPEC,
        MLP_REGRESSOR_SPEC,
        TABPFN_SPEC,
    ]
}


def get_models_for_task(task_type: str) -> list[ModelSpec]:
    """Get all models that support a given task type.

    Args:
        task_type: One of 'binary_classification', 'multiclass_classification', 'regression'

    Returns:
        List of ModelSpec objects supporting the task type.
    """
    return [spec for spec in ALL_MODELS.values() if task_type in spec.task_types]


def get_model_spec(model_name: str) -> ModelSpec:
    """Get the spec for a specific model by name.

    Raises:
        ValueError: If model name is not found in registry.
    """
    if model_name not in ALL_MODELS:
        available = ", ".join(sorted(ALL_MODELS.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return ALL_MODELS[model_name]


def list_available_models(task_type: str | None = None) -> list[dict[str, Any]]:
    """List available models, optionally filtered by task type.

    Returns:
        List of model info dicts with name, description, hyperparameter spaces.
    """
    if task_type:
        models = get_models_for_task(task_type)
    else:
        models = list(ALL_MODELS.values())
    return [m.to_dict() for m in models]


def instantiate_model(
    model_name: str,
    task_type: str,
    hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Any:
    """Create a model instance from the registry.

    Args:
        model_name: Name of the model in the registry.
        task_type: Task type to determine classifier vs regressor variant.
        hyperparameters: Override default hyperparameters.
        random_state: Random seed for reproducibility.

    Returns:
        Instantiated sklearn-compatible model.
    """
    spec = get_model_spec(model_name)

    if task_type not in spec.task_types:
        raise ValueError(f"Model '{model_name}' does not support task type '{task_type}'")

    params = spec.get_default_params()
    if hyperparameters:
        params = {**params, **hyperparameters}

    # Determine the correct class (classifier vs regressor)
    cls = _resolve_model_class(spec, task_type)

    # Add random_state if the model supports it
    init_params = _filter_valid_params(cls, params)
    if _accepts_param(cls, "random_state"):
        init_params["random_state"] = random_state

    # Model-specific adjustments
    if model_name == "tabpfn":
        # TabPFN does not accept random_state in __init__
        init_params.pop("random_state", None)
    elif model_name == "xgboost":
        init_params["verbosity"] = 0
        init_params["use_label_encoder"] = False
        if task_type == "multiclass_classification":
            init_params["objective"] = "multi:softprob"
    elif model_name == "lightgbm":
        init_params["verbose"] = -1
    elif model_name == "catboost":
        init_params["verbose"] = 0
    elif model_name == "svc":
        init_params["probability"] = True
    elif model_name == "logistic_regression":
        if params.get("penalty") == "elasticnet" and params.get("solver") != "saga":
            init_params["solver"] = "saga"

    return cls(**init_params)


def _resolve_model_class(spec: ModelSpec, task_type: str) -> type:
    """Resolve the actual model class, swapping classifier for regressor as needed.

    Raises:
        ImportError: If a third-party package (e.g. tabpfn) is not installed.
    """
    class_path = spec.sklearn_class

    is_regression = task_type == "regression"

    # TabPFN only supports classification; no classifier/regressor swapping needed
    if spec.name == "tabpfn":
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TabPFN not installed. Install with: pip install tabpfn"
            ) from exc
        return getattr(module, class_name)

    # Map classifier class paths to regressor equivalents
    if is_regression:
        replacements = {
            "XGBClassifier": "XGBRegressor",
            "LGBMClassifier": "LGBMRegressor",
            "CatBoostClassifier": "CatBoostRegressor",
            "RandomForestClassifier": "RandomForestRegressor",
            "ExtraTreesClassifier": "ExtraTreesRegressor",
        }
        for clf, reg in replacements.items():
            class_path = class_path.replace(clf, reg)

    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _accepts_param(cls: type, param_name: str) -> bool:
    """Check if a model class accepts a given parameter."""
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return True  # If we can't inspect, assume it accepts it


def _filter_valid_params(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """Filter parameters to only those accepted by the model class."""
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_kwargs:
            return dict(params)
        valid_names = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in params.items() if k in valid_names}
    except (ValueError, TypeError):
        return dict(params)
