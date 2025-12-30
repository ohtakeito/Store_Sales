import warnings
warnings.filterwarnings('ignore')

# 基本ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb


# モデル関連
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 前処理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# モデル評価・検証
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_rel
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import FunctionTransformer
import optuna
import numpy as np
from catboost import CatBoostClassifier

# パイプライン
from sklearn.pipeline import Pipeline

# 可視化フォント
import matplotlib.font_manager as font_manager
import matplotlib.font_manager as fm

# MLflow
import mlflow
import mlflow.sklearn

# その他
from functools import partial

def set_visual_style(
    font_path="C:/Windows/Fonts/YuGothL.ttc",  # Yu Gothic Light
    style="whitegrid",
    context="notebook",
    font_size=12
):
    """
    日本語フォントと seaborn スタイルを一括適用する。

    Parameters:
    ------------
    font_path : str
        使用したい日本語フォントファイル（.ttf or .ttc）のパス
    style : str
        seabornのスタイル（"whitegrid", "darkgrid", "ticks" など）
    context : str
        seabornのコンテキスト（"notebook", "paper", "talk", "poster"）
    font_size : int
        ベースのフォントサイズ
    """

    # seabornスタイルを先に設定（これが色を上書きすることがある）
    sns.set(style=style, context=context)

    # フォント指定
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = font_size

    # すべてのテキスト要素に「完全な黒」を適用
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#000000'
    plt.rcParams['ytick.color'] = '#000000'

    print(f"✅ フォント設定: {font_name}, スタイル: {style}, サイズ: {font_size}, 色: #000000")

