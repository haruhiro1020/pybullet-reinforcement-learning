# 複数ファイルで使用する定数の定義
from enum import Enum
from enum import auto


# 次元数を定義
DIMENTION_NONE  = -1    # 未定義
DIMENTION_2D    =  2    # 2次元
DIMENTION_3D    =  3    # 3次元
DIMENTION_6D    =  6    # 6次元

# 重力加速度 [m/s²]
GRAVITY_VALUE   = 9.81

# 0割を防ぐための定数
EPSILON         = 1e-6

# フォルダ名
URDF_FOLDER_NAME = "URDF"       # 全URDFを一つにまとめたフォルダ名
RESULT_FOLDER_NAME = "Result"   # 学習フェーズ内の結果を保存するフォルダ


# 補間方法の定義
class INTERPOLATION(Enum):
    """
    補間方法
    """
    NONE      = "none"      # 未定義
    JOINT     = "joint"     # 関節補間
    POSITION  = "pos"       # 位置補間


# ロボットアームが保存されている URDF ファイル名
class ROBOTURDF(Enum):
    """
    ロボットのURDFファイル名
    """
    # 2軸ロボットアーム
    DOF2 = f"{URDF_FOLDER_NAME}/robot_2dof.urdf"            # ハンド(グリッパ)なし
    DOF2_HAND = f"{URDF_FOLDER_NAME}/robot_2dof_hand.urdf"  # ハンド(グリッパ)付き


# カメラが保存されている URDF ファイル名
class CAMERAURDF(Enum):
    """
    カメラのURDFファイル名
    """
    # X軸方向に向いているカメラ
    RIGHT2LEFT = f"{URDF_FOLDER_NAME}/camera_right_to_left.urdf"    # 左向き(-X方向)のカメラ
    LEFT2RIGHT = f"{URDF_FOLDER_NAME}/camera_left_to_right.urdf"    # 右向き(+X方向)のカメラ

    # Y軸方向に向いているカメラ
    FRONT2BACK = f"{URDF_FOLDER_NAME}/camera_front_to_back.urdf"    # 奥向き(+Y方向)のカメラ
    BACK2FRONT = f"{URDF_FOLDER_NAME}/camera_back_to_front.urdf"    # 手前向き(-Y方向)のカメラ

    # Z軸方向に向いているカメラ
    UP2DOWN    = f"{URDF_FOLDER_NAME}/camera_up_to_down.urdf"       # 下向き(-Z方向)のカメラ


# カメラ数を定義する定数
class CAMERANUM(Enum):
    """
    カメラ数
    """
    SINGLE = 1          # 1つのカメラ
    MULTI  = 5          # 複数のカメラ


# 環境URDFに関する定数
class ENVURDF(Enum):
    """
    環境のURDFに関する定数
    """
    TWODOF   = f"{URDF_FOLDER_NAME}/environment_2dof.urdf"      # 2軸ロボットアーム


# 強化学習の環境に関する定数
class ENV(Enum):
    """
    強化学習の環境に関する定数
    """
    TWODOF = 0              # 2軸ロボットアーム


# 強化学習のエージェントを定義する定数
class AGENT(Enum):
    """
    強化学習のエージェント
    """
    RANDOM = 0              # ランダムに行動するエージェント
    MONTECARLO = auto()     # モンテカルロ法に沿って行動するエージェント


# 強化学習のフェーズを定義する定数
class REIN_PHASE(Enum):
    """
    強化学習のフェーズ
    """
    LEARN = 0               # 学習フェーズ
    IMITATION = auto()      # 学習の再現フェーズ
    INFERENCE = auto()      # 学習後の推論フェーズ




