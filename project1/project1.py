"""
EECS 445 Winter 2025

This script should contain most of the work for the project. You will need to fill in every TODO comment.
"""


import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


import helper


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)


challenge_on = True


def challenge_generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    df_replaced = df.replace(-1, np.nan)
    static, timeseries = df_replaced.iloc[0:5], df.iloc[5:]

    feature_dict = {} # final return dictionary
    timeseries_agg = {} # aggregate data
    
    ICUTypes = {1: "ICUType_CCU", 2: "ICUType_CSRU", 3: "ICUType_MICU", 4: "ICUType_SICU"}
    Gender = {0: "Female", 1: "Male"}
    interactive_terms = ["FiO2_SaO2", "HR_MAP", "BUN_Creatinine", "Glucose_Lactate", "Temp_WBC"]

    # put all data into timeseries_agg
    for var in timeseries_variables:
        timeseries_agg[var] = {"first_half": [], "last_half": []}        
    for _, row in timeseries.iterrows():
        time, var, val = row["Time"], row["Variable"], row["Value"]        
        if var in timeseries_variables:
            if int(time[0:2]) < 24:
                timeseries_agg[var]["first_half"].append(val)
            else:
                timeseries_agg[var]["last_half"].append(val)            

    # set static variables in feature_dict
    for _, val in ICUTypes.items():
        feature_dict[val] = 0
    for _, val in Gender.items():
        feature_dict[val] = 0
    for _, row in static.iterrows():
        var, val = row["Variable"], row["Value"]
        if var == "ICUType":
            if val == 1 or val == 2 or val == 3 or val == 4:
                feature_dict[ICUTypes[val]] = 1
        elif var == "Gender":
            if val == 0 or val == 1:
                feature_dict[Gender[val]] = 1
        else:
            feature_dict[var] = val

    # set timeseries variables in feature_dict            
    for var, val in timeseries_agg.items():
        fh, lh = np.array(val["first_half"]), np.array(val["last_half"])

        def compute_agg(arr):
            return {
                "max": np.nan, "min": np.nan, "median": np.nan, "mean": np.nan, "std": np.nan
            } if arr.size == 0 else {
                "max": np.max(arr), "min": np.min(arr), "median": np.median(arr), "mean": np.mean(arr), "std": np.std(arr)
            }
        
        for suffix, data in [("_fh", fh), ("_lh", lh)]:
            agg = compute_agg(data)
            for stat, value in agg.items():
                feature_dict[f"{var}_{stat}{suffix}"] = value
        
    # add interaction features
    for term in interactive_terms:
        f1, f2 = term.split("_")
        for stat in ["mean"]:
            for suffix in ["_fh", "_lh"]:
                f1_val = feature_dict.get(f"{f1}_{stat}{suffix}", np.nan)
                f2_val = feature_dict.get(f"{f2}_{stat}{suffix}", np.nan)
                feature_dict[f"{term}_{stat}{suffix}"] = f1_val * f2_val if not np.isnan(f1_val) and not np.isnan(f2_val) else np.nan
                        
    return feature_dict


def challenge_impute_missing_values(X: npt.NDArray, n_neighbors: int = 5) -> npt.NDArray:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(X)


def challenge():
    X_train, y_train, X_challenge, feature_names = helper.get_challenge_data()

    num_neg = sum(y_train == -1)
    num_pos = sum(y_train == 1)
    Wp = num_neg / num_pos

    param_grid = {
        "C": [0.01, 0.1, 1], 
        "penalty": ['l2', 'l1'],
        "class_weight": [{-1: 1, 1: Wp}, {-1: 1, 1: 1}],
    }

    log_reg = LogisticRegression(solver="liblinear", fit_intercept=False, random_state=42)
    
    clf = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)

    params, score = clf.best_params_, clf.best_score_
    y_pred = clf.predict(X_train)
    cm = metrics.confusion_matrix(y_train, y_pred, labels=[-1, 1])
    
    print(f"\nParameters: {params}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Best CV F1 Score: {score:.4f}")

    print(f"auroc: {metrics.roc_auc_score(y_train, clf.decision_function(X_train))}")
    print(f"avg precision: {metrics.average_precision_score(y_train, clf.decision_function(X_train))}")

    TN, FP, FN, TP = cm.ravel()
    print(f"accuracy: {(TP + TN) / (TP + FP + TN + FN)}")
    print(f"precision: {TP / (TP + FP) if (TP + FP) != 0 else 0}")
    print(f"sensitivity: {TP / (TP + FN) if (TP + FN) != 0 else 0}")
    print(f"specificity:{ TN / (TN + FP) if (TN + FP) != 0 else 0}")
    
    helper.save_challenge_predictions(y_label=clf.predict(X_challenge).astype(int), y_score=clf.decision_function(X_challenge), uniqname="alxyang")


def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    
    if challenge_on:
        return challenge_generate_feature_vector(df)
    
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df_replaced = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df_replaced.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    for var in timeseries_variables:
        feature_dict["max_" + var] = np.nan

    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for _, row in static.iterrows():
        var, val = row["Variable"], row["Value"]
        feature_dict[var] = val

    # TODO  3) extract max of time-varying variables into feature dict
    for _, row in timeseries.iterrows():
        if row["Variable"] in timeseries_variables:
            var, val = "max_" + row["Variable"], row["Value"]
            if np.isnan(feature_dict[var]):
                feature_dict[var] = val
            else:
                feature_dict[var] = np.maximum(feature_dict[var], val)
            
    return feature_dict


def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: array of shape (N, d) which could contain missing values
        
    Returns:
        X: array of shape (N, d) without missing values
    """
    
    if challenge_on:
        return challenge_impute_missing_values(X)
    
    for col in range(X.shape[1]):  # Iterate over each column
        mean_val = np.nanmean(X[:, col])  # Compute mean for that column
        X[np.isnan(X[:, col]), col] = mean_val  # Replace NaNs with column mean

    return X


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: array of shape (N, d).

    Returns:
        X: array of shape (N, d). Values are normalized per column.
    """
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    den = col_max - col_min
    den[den == 0] = 1  # what if min == max?

    return (X - col_min) / den


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver="liblinear", fit_intercept=False, random_state=seed)
    elif loss == "squared_error":
        return KernelRidge(alpha=1 / (2 * C), kernel=kernel, gamma=gamma)


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy"
) -> float:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.

    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
    Returns:
        peformance for the specific metric
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    # TODO: implement

    y_pred = clf_trained.predict(X)
    
    if metric == "auroc":
        if hasattr(clf_trained, 'decision_function'):
            return metrics.roc_auc_score(y_true, clf_trained.decision_function(X))
        else:
            return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        if hasattr(clf_trained, 'decision_function'):
            return metrics.average_precision_score(y_true, clf_trained.decision_function(X))
        else:
            return metrics.average_precision_score(y_true, y_pred)
    else:
        y_pred = np.where(y_pred >= 0, 1, -1)
        cm = metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1])
        TN, FP, FN, TP = cm.ravel()
    
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    
        if metric == "accuracy":
            return accuracy
        elif metric == "precision":
            return precision
        elif metric == "sensitivity":
            return sensitivity
        elif metric == "specificity":
            return specificity
        # f1-score:
        return (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        

def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    # TODO: implement

    skf = StratifiedKFold(n_splits=k)
    performances = []

    for training, validation in skf.split(X, y):
        X_train, X_val = X[training], X[validation]
        y_train, y_val = y[training], y[validation]

        clf.fit(X_train, y_train)
        performances.append(performance(clf, X_val, y_val, metric))

    return (np.mean(performances), np.min(performances), np.max(performances))


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    penalties: list[str] = ["l1", "l2"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # NOTE: use your cv_performance function to evaluate the performance of each classifier
    # TODO: implement
    
    best = None
    best_perf = -float('inf')

    for C in C_range:        
        for penalty in penalties:
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            mean_perf, minny, maxxy = cv_performance(clf, X, y, metric=metric, k=k)
            if mean_perf > best_perf:
                best_perf = mean_perf
                best = (C, penalty)
            
            # TO see how performance varies with C, 2c
            # print(C, penalty, mean_perf, minny, maxxy)
    return best


def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    # TODO: implement
    
    best = None
    best_perf = -float('inf')
    
    for C in C_range:
        for gamma in gamma_range:
            clf = get_classifier(loss="squared_error", C=C, gamma=gamma)
            mean_perf, minny, maxxy = cv_performance(clf, X, y, metric=metric, k=k)           
            if mean_perf > best_perf:
                best_perf = mean_perf
                best = (C, gamma)
                
            if C == 1:
                print(gamma, mean_perf, minny, maxxy)

    return best


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}

    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            
            # TODO: fit clf to X and y
            clf.fit(X, y)
            
            # TODO: extract learned coefficients from clf into w
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_.flatten()
            
            # TODO: count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def print_feature_summary_1d(X_train, feature_names) -> None:
    df_train = pd.DataFrame(X_train, columns=feature_names)
    summary_table = pd.DataFrame({
        "Feature": feature_names,
        "Mean": df_train.mean(),
        "IQR": df_train.quantile(0.75) - df_train.quantile(0.25)
    })
    print(summary_table.to_string(index=False))


def print_best_cv_performance_2c(X_train, y_train, metric_list) -> None:
    results = []
    for metric in metric_list:
        best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric, C_range=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        clf = LogisticRegression(penalty=best_penalty, C=best_C, solver="liblinear", fit_intercept=False, random_state=seed)
        mean_perf, min_perf, max_perf = cv_performance(clf, X_train, y_train, metric=metric)
        results.append({
            "Metric": metric,
            "Best C": best_C,
            "Penalty": best_penalty, 
            "CV Performance": "" + str(mean_perf) + " (" + str(min_perf) + ", " + str(max_perf) + ")",
        })
        
    print(f"{'Metric':<20}{'Best C':<10}{'Penalty':<10}{'CV Performance':<20}")
    print("="*100)
    for res in results:
        print(f"{res['Metric']:<20}{res['Best C']:<10}{res['Penalty']:<10}{res['CV Performance']:<20}")
    
        
def print_performance_2d(X_train, y_train, X_test, y_test, metric_list) -> None:
    clf = LogisticRegression(penalty='l1', C=1, solver="liblinear", fit_intercept=False, random_state=seed)
    clf.fit(X_train, y_train)
    for metric in metric_list:
        score = performance(clf, X_test, y_test, metric)
        print(f"{metric}: {score}")


def coefficient_2f(X_train, y_train, feature_names) -> None:
    clf = LogisticRegression(penalty='l1', C=1, solver="liblinear", fit_intercept=False, random_state=seed)
    clf.fit(X_train, y_train)
    coefficients = clf.coef_.flatten()
    top_positive_indices = np.argsort(coefficients)[-4:][::-1]  # Sort and take top 4 in descending order
    top_negative_indices = np.argsort(coefficients)[:4]  # Take bottom 4 in ascending order

    # Print results in Table 3 format
    print("positive")
    for i, idx in enumerate(top_positive_indices):
        print(feature_names[idx], coefficients[idx])

    print("negative")
    for i, idx in enumerate(top_negative_indices):
        print(feature_names[idx], coefficients[idx])


def class_weight_performance_3b(X_train, y_train, X_test, y_test, metric_list) -> None:
    clf = get_classifier(loss="logistic", penalty="l2", C=1, class_weight={-1: 1, 1: 50})
    clf.fit(X_train, y_train)
    
    for metric in metric_list:
        score = performance(clf, X_test, y_test, metric)
        print(f"{metric}: {score}")


def select_class_weight_3a(X_train, y_train, X_test, y_test, metric_list, metric="f1-score", k=5) -> None:
    # def tune_class_weights(X, y, class_weight_range, metric, k=5, seed=42):
    best = None
    best_perf = -float('inf')

    num_neg = sum(y_train == -1)
    num_pos = sum(y_train == 1)
    Wp = num_neg / num_pos
    Wn = 1
    
    scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for scale in scales:        
        weights = [
            {-1: Wn, 1: Wp},
            {-1: Wn + 1*scale, 1: Wp},
            {-1: Wn + 2*scale, 1: Wp},
            {-1: Wn + 3*scale, 1: Wp},
            {-1: Wn, 1: Wp + 1*scale},
            {-1: Wn, 1: Wp + 2*scale},
            {-1: Wn, 1: Wp + 3*scale},
        ]
        normalized = [
            {key: weight / min(w[-1], w[1]) for key, weight in w.items()}
            for w in weights
        ]
                
        for weight in normalized:
            clf = get_classifier(loss="logistic", penalty="l2", C=1, class_weight=weight)
            mean_perf, _, _ = cv_performance(clf, X_train, y_train, metric=metric, k=k)
            if mean_perf > best_perf:
                best_perf = mean_perf
                best = weight
        
    print("Best weight: ", best, best_perf)
        
    clf = get_classifier(loss="logistic", penalty="l2", C=1, class_weight=best)
    clf.fit(X_train, y_train)
    
    for m in metric_list:
        score = performance(clf, X_test, y_test, m)
        print(f"{m}: {score}")
    

def plot_roc_curves(X_train, y_train, X_test, y_test):
    weights = [
        ({-1: 1, 1: 1}, "Wn=1, Wp=1"),
        ({-1: 1, 1: 5}, "Wn=1, Wp=5")
    ]

    plt.figure(figsize=(8, 6))

    for weight, label in weights:
        clf = get_classifier(loss="logistic", penalty="l2", C=1, class_weight=weight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (auc = {roc_auc:.2f})")
        
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig("ROC_curves.png", dpi=200)
    plt.close()

    
def logreg_vs_kernridg(X_train, y_train, X_test, y_test, metric_list):
    C = 1
    clf_log = LogisticRegression(penalty='l2', C=C, fit_intercept=False, random_state=seed)
    clf_kern = KernelRidge(alpha=1/(2*C), kernel="linear")
    
    clf_log.fit(X_train, y_train)
    clf_kern.fit(X_train, y_train)
    
    for m in metric_list:
        score = performance(clf_log, X_test, y_test, m)
        print(f"{m}: {score}")
        
    for m in metric_list:
        score = performance(clf_kern, X_test, y_test, m)
        print(f"{m}: {score}")


def rbf_auroc_4b(X_train, y_train, X_test, y_test, metric_list):
    best_C, best_gamma = select_param_RBF(X_train, y_train, metric="auroc", C_range=[0.01, 0.1, 1.0, 10, 100], gamma_range=[0.01, 0.1, 1, 10])
    print("BEST:", best_C, best_gamma)
    clf = get_classifier(loss="squared_error", C=best_C, gamma=best_gamma)
    clf.fit(X_train, y_train)

    for m in metric_list:
        score = performance(clf, X_test, y_test, m)
        print(f"{m}: {score}")


def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: Only set debug=True when testing your implementation against debug.txt. DO NOT USE debug=True when
    #       answering the project questions! It only loads a small sample (n = 100) of the data in debug mode,
    #       so your performance will be very bad when working with the debug data.
    # X_train, y_train, X_test, y_test, feature_names = helper.get_project_data(debug=False)

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    # print_feature_summary_1d(X_train, feature_names)
    # print_best_cv_performance_2c(X_train, y_train, metric_list)
    # print_performance_2d(X_train, y_train, X_test, y_test, metric_list)
    # plot_weight(X_train, y_train, [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], ["l1", "l2"])
    # coefficient_2f(X_train, y_train, feature_names)    
    # class_weight_performance_3b(X_train, y_train, X_test, y_test, metric_list)
    # select_class_weight_3a(X_train, y_train, X_test, y_test, metric_list)
    # plot_roc_curves(X_train, y_train, X_test, y_test)
    # logreg_vs_kernridg(X_train, y_train, X_test, y_test, metric_list)
    # rbf_auroc_4b(X_train, y_train, X_test, y_test, metric_list)

    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()
    
    challenge()


if __name__ == "__main__":
    main()
