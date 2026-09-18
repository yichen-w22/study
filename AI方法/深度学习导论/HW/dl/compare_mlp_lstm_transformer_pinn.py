import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 全局配置（你已同意：所有模型用同一 PINN 数据集、同一 test 切分）
# =========================================================
DATA_PATH = r"D:\清华\毕业设计\test\dl\data\PINN选取参数_1min.csv"
MODEL_SAVE_DIR = r"D:\清华\毕业设计\test\dl\saved_models"
RESULTS_SAVE_PATH = r"D:\清华\毕业设计\test\dl\model_performance.xlsx"
TARGET_COLS = ["总负荷", "GT1实发", "GT2实发", "汽机负荷"]
SEQ_LEN = 30

EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 5
MIN_DELTA = 1e-6

# PINN 物理项权重
LAM_GT = 0.2
LAM_ST = 0.2

# 物理常数（极简）
CP_AIR = 1.005   # kJ/kg-K
CP_GAS = 1.150   # kJ/kg-K
LHV = 47.5       # MJ/kg（你指定）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# 创建保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# =========================================================
# 1) 数据预处理：沿用你 PINN 方案（数值化 + 填充 + phys 有限化）
# =========================================================
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def F_to_K(x):
    return (x - 32.0) * 5.0/9.0 + 273.15

def t_per_h_to_kg_per_s(x):
    return x * 1000.0 / 3600.0

def load_pinn_dataset(csv_path=DATA_PATH):
    data = pd.read_csv(csv_path)

    # ---- 目标 ----
    y_df = data[TARGET_COLS].apply(to_num)
    mask_y = (y_df.notna().all(axis=1)) & (y_df > 0).all(axis=1)

    data = data.loc[mask_y].reset_index(drop=True)
    y_df = y_df.loc[mask_y].reset_index(drop=True)

    # ---- 环境（中文：℃、kPa）----
    data["大气气温"] = to_num(data["大气气温"])
    data["大气气压"] = to_num(data["大气气压"])
    T_amb_K = (data["大气气温"] + 273.15).reset_index(drop=True)

    # ---- 英文温度（按华氏→K）----
    T_ci1_K  = F_to_K(to_num(data["Compressor Inlet Temperature"])).reset_index(drop=True)
    T_exh1_K = F_to_K(to_num(data["Exhaust Temp MeDXan Corrected By Average"])).reset_index(drop=True)
    T_ci2_K  = F_to_K(to_num(data["Compressor Inlet Temperature.1"])).reset_index(drop=True)
    T_exh2_K = F_to_K(to_num(data["Exhaust Temp MeDXan Corrected By Average.1"])).reset_index(drop=True)

    # ---- 锅炉出口排烟温度（中文：℃→K）----
    T_stack1_K = ((to_num(data["#1锅炉出口排烟温度1"]) +
                   to_num(data["#1锅炉出口排烟温度2"]) +
                   to_num(data["#1锅炉出口排烟温度3"])) / 3.0 + 273.15).reset_index(drop=True)

    T_stack2_K = ((to_num(data["#2锅炉出口排烟温度2"]) +
                   to_num(data["#2锅炉出口排烟温度3"])) / 2.0 + 273.15).reset_index(drop=True)

    # ---- 流量（t/h→kg/s）----
    m_f1 = t_per_h_to_kg_per_s(to_num(data["Gas Fuel Flow"])).reset_index(drop=True)
    m_g1 = t_per_h_to_kg_per_s(to_num(data["Turbine Exhaust Mass Flow"])).reset_index(drop=True)
    m_f2 = t_per_h_to_kg_per_s(to_num(data["Gas Fuel Flow.1"])).reset_index(drop=True)
    m_g2 = t_per_h_to_kg_per_s(to_num(data["Turbine Exhaust Mass Flow.1"])).reset_index(drop=True)

    # ---- 特征：除目标外全用，数值化 + 删全NaN列 + 中位数填充 ----
    feature_cols = [c for c in data.columns if c not in TARGET_COLS]
    X_df = data[feature_cols].apply(to_num)

    X_df = X_df.dropna(axis=1, how="all")
    X_df = X_df.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0.0)

    # ---- phys：同样有限化（关键：避免 loss nan）----
    phys_df = pd.DataFrame({
        "Tref":  T_amb_K,
        "Tci1":  T_ci1_K,
        "Texh1": T_exh1_K,
        "Tst1":  T_stack1_K,
        "mf1":   m_f1,
        "mg1":   m_g1,
        "Tci2":  T_ci2_K,
        "Texh2": T_exh2_K,
        "Tst2":  T_stack2_K,
        "mf2":   m_f2,
        "mg2":   m_g2,
    }).apply(to_num).replace([np.inf, -np.inf], np.nan)

    phys_df = phys_df.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0.0)

    X = X_df.to_numpy(np.float32)
    y = y_df.to_numpy(np.float32)
    phys = phys_df.to_numpy(np.float32)

    return X, y, phys, X_df.columns.tolist()


# =========================================================
# 2) 统一切分（一次切分，所有模型共用）
# =========================================================
def time_split(X_scaled, y_scaled, phys, test_size=0.2, val_frac_in_trainval=0.125):
    X_trainval, X_test, y_trainval, y_test, phys_trainval, phys_test = train_test_split(
        X_scaled, y_scaled, phys, test_size=test_size, shuffle=False
    )
    X_train, X_val, y_train, y_val, phys_train, phys_val = train_test_split(
        X_trainval, y_trainval, phys_trainval, test_size=val_frac_in_trainval, shuffle=False
    )
    return (X_train, y_train, phys_train,
            X_val, y_val, phys_val,
            X_test, y_test, phys_test)


# =========================================================
# 3) 序列构造（给 LSTM / Transformer）
#    每个样本对应"预测最后一个时刻的 y"
# =========================================================
def make_sequences(X_scaled, y_scaled, seq_len=SEQ_LEN):
    N = len(X_scaled)
    X_seq = []
    y_seq = []
    for t in range(seq_len - 1, N):
        X_seq.append(X_scaled[t - seq_len + 1: t + 1])
        y_seq.append(y_scaled[t])
    return np.asarray(X_seq, np.float32), np.asarray(y_seq, np.float32)


# =========================================================
# 4) 通用训练函数（早停 + 梯度裁剪 + 保存最优模型）
# =========================================================
def train_model(model, model_name, train_loader, X_val, y_val, loss_fn, optimizer,
                epochs=EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA):
    best_val = float("inf")
    best_state = None
    bad = 0

    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu().item()) * len(batch[0])
            n += len(batch[0])

        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = torch.mean((val_pred - y_val) ** 2).item()

        print(f"Epoch {ep:04d} TrainLoss: {train_loss:.6f} | ValLoss: {val_loss:.6f}")

        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f"[EarlyStop] epoch={ep}, best_val_loss={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        # 保存最优模型
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pth")
        torch.save({
            'model_state_dict': best_state,
            'val_loss': best_val,
            'epoch': ep - patience  # 减去早停的epoch数
        }, model_save_path)
        print(f"Saved best {model_name} model to {model_save_path}")
    
    return model


# =========================================================
# 5) 四个模型定义
# =========================================================
# ---- MLP ----
class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# ---- LSTM ----
class LSTMNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # (B, H)
        return self.fc(last)           # (B, out_dim)

# ---- Transformer ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.fc(last)

# ---- PINN（MLP 主干 + 物理参数）----
class PINNNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, out_dim)

        self.b_gt1 = nn.Parameter(torch.zeros(1))
        self.b_gt2 = nn.Parameter(torch.zeros(1))
        self.b_st  = nn.Parameter(torch.zeros(1))
        self.k_cond_raw = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4(h)

    def k_cond(self):
        return torch.nn.functional.softplus(self.k_cond_raw)


# =========================================================
# 6) PINN loss：data + GT 能量 + ST 能量（极简、稳健）
# =========================================================
def mw_from_mcpdt(m, cp_kj, dT):
    return (m * cp_kj * dT) / 1000.0  # kW->MW

def pinn_loss_fn(net, y_pred_scaled, y_true_scaled, phys_batch, y_scaler):
    # 监督项（scaled）
    L_data = torch.mean((y_pred_scaled - y_true_scaled) ** 2)

    # 反归一化到 MW（更稳：y_real = y_scaled/scale + data_min）
    y_scale_t = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=DEVICE)
    y_min_t   = torch.tensor(y_scaler.data_min_, dtype=torch.float32, device=DEVICE)
    y_pred = y_pred_scaled / y_scale_t + y_min_t

    P_gt1 = y_pred[:, 1]
    P_gt2 = y_pred[:, 2]
    P_st  = y_pred[:, 3]

    # phys: [Tref,Tci1,Texh1,Tst1,mf1,mg1,Tci2,Texh2,Tst2,mf2,mg2]
    Tref  = phys_batch[:, 0]
    Tci1  = phys_batch[:, 1]
    Texh1 = phys_batch[:, 2]
    Tst1  = phys_batch[:, 3]
    mf1   = phys_batch[:, 4]
    mg1   = phys_batch[:, 5]
    Tci2  = phys_batch[:, 6]
    Texh2 = phys_batch[:, 7]
    Tst2  = phys_batch[:, 8]
    mf2   = phys_batch[:, 9]
    mg2   = phys_batch[:,10]

    ma1 = torch.clamp(mg1 - mf1, min=0.0)
    ma2 = torch.clamp(mg2 - mf2, min=0.0)

    Ein_air1  = mw_from_mcpdt(ma1, CP_AIR, (Tci1 - Tref))
    Ein_air2  = mw_from_mcpdt(ma2, CP_AIR, (Tci2 - Tref))
    Ein_fuel1 = mf1 * LHV
    Ein_fuel2 = mf2 * LHV
    Eout_exh1 = mw_from_mcpdt(mg1, CP_GAS, (Texh1 - Tref))
    Eout_exh2 = mw_from_mcpdt(mg2, CP_GAS, (Texh2 - Tref))

    r_gt1 = (Ein_air1 + Ein_fuel1) - (Eout_exh1 + P_gt1 + net.b_gt1)
    r_gt2 = (Ein_air2 + Ein_fuel2) - (Eout_exh2 + P_gt2 + net.b_gt2)

    Qdrop = mw_from_mcpdt(mg1, CP_GAS, (Texh1 - Tst1)) + mw_from_mcpdt(mg2, CP_GAS, (Texh2 - Tst2))
    kcond = net.k_cond()
    r_st = Qdrop - ((1.0 + kcond) * P_st + net.b_st)

    # 稳健：裁剪 + nan_to_num
    r_gt1 = torch.nan_to_num(torch.clamp(r_gt1, -5e4, 5e4), nan=0.0, posinf=0.0, neginf=0.0)
    r_gt2 = torch.nan_to_num(torch.clamp(r_gt2, -5e4, 5e4), nan=0.0, posinf=0.0, neginf=0.0)
    r_st  = torch.nan_to_num(torch.clamp(r_st,  -5e4, 5e4), nan=0.0, posinf=0.0, neginf=0.0)

    L_gt = torch.mean((r_gt1/1000.0)**2 + (r_gt2/1000.0)**2)
    L_st = torch.mean((r_st/1000.0)**2)

    return L_data + LAM_GT * L_gt + LAM_ST * L_st


# =========================================================
# 7) 训练+预测：四个模型都用同一 test（同一时间段）
# =========================================================
def fit_predict_mlp(X_train, y_train, X_val, y_val, X_test):
    model = MLPNet(X_train.shape[1], y_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def batch_loss(m, batch):
        xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
        pred = m(xb)
        return torch.mean((pred - yb)**2)

    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    model = train_model(model, "mlp", train_loader, Xv, yv, batch_loss, opt)

    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return yp

def fit_predict_pinn(X_train, y_train, phys_train, X_val, y_val, phys_val, X_test, phys_test, y_scaler):
    model = PINNNet(X_train.shape[1], y_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(phys_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def batch_loss(m, batch):
        xb = batch[0].to(DEVICE)
        yb = batch[1].to(DEVICE)
        pb = batch[2].to(DEVICE)
        pred = m(xb)
        return pinn_loss_fn(m, pred, yb, pb, y_scaler)

    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    model = train_model(model, "pinn", train_loader, Xv, yv, batch_loss, opt)

    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return yp

def fit_predict_lstm(Xseq_train, y_train, Xseq_val, y_val, Xseq_test, hidden_dim=64):
    model = LSTMNet(Xseq_train.shape[2], hidden_dim, y_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.tensor(Xseq_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def batch_loss(m, batch):
        xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
        pred = m(xb)
        return torch.mean((pred - yb)**2)

    Xv = torch.tensor(Xseq_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    model = train_model(model, "lstm", train_loader, Xv, yv, batch_loss, opt)

    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(Xseq_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return yp

def fit_predict_transformer(Xseq_train, y_train, Xseq_val, y_val, Xseq_test):
    model = TransformerRegressor(Xseq_train.shape[2], y_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ds = TensorDataset(torch.tensor(Xseq_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def batch_loss(m, batch):
        xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
        pred = m(xb)
        return torch.mean((pred - yb)**2)

    Xv = torch.tensor(Xseq_val, dtype=torch.float32).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    model = train_model(model, "transformer", train_loader, Xv, yv, batch_loss, opt)

    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(Xseq_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return yp


# =========================================================
# 8) 保存性能指标到Excel
# =========================================================
def save_performance_to_excel(performance_data, save_path):
    """保存性能指标到Excel文件"""
    
    # 创建DataFrame
    rows = []
    for model_name, metrics in performance_data.items():
        for target_name, metric_dict in metrics.items():
            rows.append({
                '模型': model_name,
                '目标变量': target_name,
                'MAE': metric_dict['MAE'],
                'RMSE': metric_dict['RMSE'],
                'R2': metric_dict['R2']
            })
    
    df = pd.DataFrame(rows)
    
    # 保存到Excel
    df.to_excel(save_path, index=False, float_format='%.4f')
    print(f"\n性能指标已保存到: {save_path}")
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("模型性能汇总:")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


# =========================================================
# 9) 主程序：统一数据 -> 统一切分 -> 统一 test -> 同图对比
# =========================================================
def main():
    # ---- load ----
    X, y, phys, feat_cols = load_pinn_dataset(DATA_PATH)
    print("[INFO] loaded:", X.shape, y.shape, phys.shape)

    # ---- scale（统一）----
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y).astype(np.float32)

    # ---- split（统一）----
    (X_tr, y_tr, phys_tr,
     X_va, y_va, phys_va,
     X_te, y_te, phys_te) = time_split(X_scaled, y_scaled, phys)

    # ---- 序列（在整个数据上先做，再用同一边界切）----
    Xseq_all, yseq_all = make_sequences(X_scaled, y_scaled, seq_len=SEQ_LEN)

    # 序列样本 t 对应原始时刻 idx = t + (SEQ_LEN-1)
    # 因此：原始切分边界要映射到序列样本边界
    N = len(X_scaled)
    n_test = int(np.floor(0.2 * N))
    test_start = N - n_test
    n_val = int(np.floor(0.125 * (N - n_test)))
    val_start = (N - n_test) - n_val  # trainval 末尾切 val

    # 序列的可用时刻从 (SEQ_LEN-1) 开始
    seq_offset = SEQ_LEN - 1
    # 把原始边界映射到序列索引
    seq_train_end = max(0, val_start - seq_offset)
    seq_val_end   = max(0, test_start - seq_offset)

    Xseq_tr = Xseq_all[:seq_train_end]
    yseq_tr = yseq_all[:seq_train_end]

    Xseq_va = Xseq_all[seq_train_end:seq_val_end]
    yseq_va = yseq_all[seq_train_end:seq_val_end]

    Xseq_te = Xseq_all[seq_val_end:]
    yseq_te = yseq_all[seq_val_end:]

    # ---- 确保所有模型使用相同的测试时间段 ----
    # 对于点模型（MLP/PINN）：从test_start开始，但要和序列模型对齐长度
    # 序列模型的测试集长度：len(Xseq_te)
    # 点模型需要从X_te中取相同数量的样本
    
    seq_test_len = len(Xseq_te)
    
    # 从X_te中取最后seq_test_len个样本
    X_te_aligned = X_te[-seq_test_len:]
    y_te_aligned = y_te[-seq_test_len:]
    phys_te_aligned = phys_te[-seq_test_len:]
    
    print(f"[INFO] 序列模型测试集长度: {seq_test_len}")
    print(f"[INFO] 点模型测试集对齐后长度: {len(X_te_aligned)}")
    print(f"[INFO] Xseq_te shape: {Xseq_te.shape}")
    print(f"[INFO] X_te_aligned shape: {X_te_aligned.shape}")

    # 对应的真实值（scaled + real）
    y_test_real = y_scaler.inverse_transform(y_te_aligned)

    # ---- 训练 + 预测（四个模型）----
    print("\n" + "="*60)
    print("开始训练 MLP 模型")
    print("="*60)
    ypred_mlp = fit_predict_mlp(X_tr, y_tr, X_va, y_va, X_te_aligned)

    print("\n" + "="*60)
    print("开始训练 PINN 模型")
    print("="*60)
    ypred_pinn = fit_predict_pinn(X_tr, y_tr, phys_tr, X_va, y_va, phys_va, 
                                  X_te_aligned, phys_te_aligned, y_scaler)

    print("\n" + "="*60)
    print("开始训练 LSTM 模型")
    print("="*60)
    ypred_lstm = fit_predict_lstm(Xseq_tr, yseq_tr, Xseq_va, yseq_va, Xseq_te)

    print("\n" + "="*60)
    print("开始训练 Transformer 模型")
    print("="*60)
    ypred_trans = fit_predict_transformer(Xseq_tr, yseq_tr, Xseq_va, yseq_va, Xseq_te)

    # ---- 反归一化（统一）----
    ypred_mlp_real  = y_scaler.inverse_transform(np.nan_to_num(ypred_mlp))
    ypred_pinn_real = y_scaler.inverse_transform(np.nan_to_num(ypred_pinn))
    ypred_lstm_real = y_scaler.inverse_transform(np.nan_to_num(ypred_lstm))
    ypred_trans_real= y_scaler.inverse_transform(np.nan_to_num(ypred_trans))

    # ---- 指标（同一 test）----
    def compute_metrics(y_true, y_pred):
        metrics = {}
        for i, tname in enumerate(TARGET_COLS):
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            metrics[tname] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            print(f"{tname}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        return metrics

    performance_data = {}
    
    print("\n" + "="*60)
    print("MLP 模型性能:")
    print("="*60)
    performance_data['MLP'] = compute_metrics(y_test_real, ypred_mlp_real)

    print("\n" + "="*60)
    print("PINN 模型性能:")
    print("="*60)
    performance_data['PINN'] = compute_metrics(y_test_real, ypred_pinn_real)

    print("\n" + "="*60)
    print("LSTM 模型性能:")
    print("="*60)
    performance_data['LSTM'] = compute_metrics(y_test_real, ypred_lstm_real)

    print("\n" + "="*60)
    print("Transformer 模型性能:")
    print("="*60)
    performance_data['Transformer'] = compute_metrics(y_test_real, ypred_trans_real)

    # ---- 保存性能指标到Excel ----
    save_performance_to_excel(performance_data, RESULTS_SAVE_PATH)

    # ---- 画图：每个目标一张，真实值 + 4条预测 ----
    x_axis = np.arange(len(y_test_real))

    for i, name in enumerate(TARGET_COLS):
        plt.figure(figsize=(12, 4))
        plt.plot(x_axis, y_test_real[:, i], label="真实值", linewidth=1.5)
        plt.plot(x_axis, ypred_mlp_real[:, i], label="MLP", alpha=0.8, linewidth=1)
        plt.plot(x_axis, ypred_lstm_real[:, i], label="LSTM", alpha=0.8, linewidth=1)
        plt.plot(x_axis, ypred_trans_real[:, i], label="Transformer", alpha=0.8, linewidth=1)
        plt.plot(x_axis, ypred_pinn_real[:, i], label="PINN", alpha=0.8, linewidth=1)
        plt.title(f"测试集对比 - {name}")
        plt.xlabel("测试集时间索引")
        plt.ylabel("MW")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # 保存图像
        fig_path = os.path.join(MODEL_SAVE_DIR, f"{name}_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()