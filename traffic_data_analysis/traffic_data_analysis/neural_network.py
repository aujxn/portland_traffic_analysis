import polars as pl
import numpy as np
import os, calendar, pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from traffic_data_analysis.config import DATA_DIR
from traffic_data_analysis.utils import load_directional_hourly
from traffic_data_analysis.preprocess import filter_bad_directional

model_file = DATA_DIR / "best_model_emb.pth"
cont_feature_file = DATA_DIR / "processed_features_cont.npy"
cat_feature_file = DATA_DIR / "processed_features_cat.npy"
target_file = DATA_DIR / "processed_targets.npy"
dummy_vocab_file = DATA_DIR / "dummy_vocab.pkl"
normalize_file = DATA_DIR / "cont_norm.pkl"

def preprocess():
    """
    Processes raw data from traffic and meta tables into features suitable for a neural network.
    Categorical variables are one-hot encoded using Polars' `to_dummies()`.
    
    Returns:
      X_cont: continuous features as a NumPy array of shape (n_days, 5)
      X_cat: one-hot encoded categorical features as a NumPy array of shape (n_days, 30)
      Y: targets as a NumPy array of shape (n_days, 24)
      dummy_vocab: dictionary mapping each categorical variable name to the list of dummy column names (and their positions)
      cont_norm: dictionary with (min, max) for continuous features (columns: Year, PosYear, PosMonth, Lat, Lon)
    """
    df_traffic, df_meta = load_directional_hourly()
    df_traffic = filter_bad_directional(df_traffic)
    df_traffic = df_traffic.with_columns(pl.col("DateTime").dt.date().alias("Date"))
    
    # Group by LocationID, Direction, Date and keep groups with exactly 24 rows.
    daily_records = []
    for keys, group in df_traffic.group_by(["LocationID", "Direction", "Date"]):
        if group.height != 24:
            continue
        group = group.sort("Hour")
        Y = group["Volume"].to_numpy()
        loc, dir_, date_val = keys
        year = date_val.year
        month = date_val.month
        day = date_val.day
        pos_year = (month - 1) / 11.0
        days_in_month = calendar.monthrange(year, month)[1]
        pos_month = day / days_in_month
        dow = date_val.weekday()
        
        # Get meta info: filter df_meta by LocationID
        meta_row = df_meta.filter(pl.col("LocationID") == loc).row(0, named=True)
        lat = meta_row["LAT"]
        lon = meta_row["LONGTD"]
        
        daily_records.append({
            "LocationID": str(loc),
            "Direction": str(dir_),
            "Date": date_val,
            "Year": year,
            "Month": month,
            "Day": day,
            "PosYear": pos_year,
            "PosMonth": pos_month,
            "DayOfWeek": dow,
            "Lat": lat,
            "Lon": lon,
            "Y": Y
        })
    
    daily_df = pl.DataFrame(daily_records)
    
    # Merge additional meta info: COUNTYNAME and HWYNUMB.
    df_meta = df_meta.with_columns(pl.col("LocationID").cast(pl.Utf8))
    daily_df = daily_df.join(df_meta.select(["LocationID", "COUNTYNAME", "HWYNUMB"]), on="LocationID", how="left")
    
    # --- Continuous Features ---
    # Columns: Year, PosYear, PosMonth, Lat, Lon
    cont_cols = ["Year", "PosYear", "PosMonth", "Lat", "Lon"]
    cont_df = daily_df.select(cont_cols)
    # Compute min and max for each continuous column (as Python scalars)
    cont_min = {col: cont_df.select(pl.col(col).min()).item() for col in cont_cols}
    cont_max = {col: cont_df.select(pl.col(col).max()).item() for col in cont_cols}
    cont_norm = {col: (cont_min[col], cont_max[col]) for col in cont_cols}
    
    # Normalize continuous columns: (x - min) / (max - min)
    for col in cont_cols:
        daily_df = daily_df.with_columns(((pl.col(col) - cont_min[col]) / (cont_max[col] - cont_min[col])).alias(col))
    
    X_cont = daily_df.select(cont_cols).to_numpy().astype(np.float64)
    
    # --- Categorical Features ---
    # Variables: LocationID, Direction, COUNTYNAME, HWYNUMB, DayOfWeek.
    # Cast appropriate columns to Utf8 (strings) except DayOfWeek (int).
    for col in ["LocationID", "Direction", "COUNTYNAME", "HWYNUMB"]:
        daily_df = daily_df.with_columns(pl.col(col).cast(pl.Utf8))
    
    # Use Polars' to_dummies on these columns.
    cat_cols = ["LocationID", "Direction", "COUNTYNAME", "HWYNUMB", "DayOfWeek"]
    # For DayOfWeek, we first cast it to Utf8 so dummies are created.
    daily_df = daily_df.with_columns(pl.col("DayOfWeek").cast(pl.Utf8))
    dummies_df = daily_df.select(cat_cols).to_dummies()
    
    # Create vocab dictionary that maps each original categorical column to a list of dummy column names.
    dummy_vocab = {c: i for i, c in enumerate(dummies_df.columns)}
    
    X_cat = dummies_df.to_numpy().astype(np.float64)
    
    # --- Targets ---
    # Column "Y" holds lists/arrays; we convert to a 2D numpy array by stacking.
    # We use to_series() to get a list.
    Y_list = daily_df.select("Y").to_series().to_list()
    Y_arr = np.stack(Y_list).astype(np.float64)  # shape (n_days, 24)
    
    # Optionally, save processed data
    # (Here we save as CSV for X_cont, and NumPy .npy files for X_cat and Y.)
    np.save(cont_feature_file, X_cont)
    np.save(cat_feature_file, X_cat)
    np.save(target_file, Y_arr)
    with open(dummy_vocab_file, "wb") as f:
        pickle.dump(dummy_vocab, f)
    with open(normalize_file, "wb") as f:
        pickle.dump(cont_norm, f)
    
    print("Processed data shapes:", X_cont.shape, X_cat.shape, Y_arr.shape)
    return X_cont, X_cat, Y_arr, dummy_vocab, cont_norm

# -------------------------------
# 2. Inference Mapping Function (Pure Polars)
# -------------------------------
def map_input_feature(location_id, direction, date, df_meta, cont_norm, dummy_vocab):
    """
    Given a location_id, direction, and a date string ('yyyy-mm-dd'),
    returns a tuple (X_cont, X_cat) as torch tensors.
    - X_cont: 5-dimensional continuous feature vector (normalized)
    - X_cat: one-hot encoded categorical vector (using dummy_vocab)
    
    All processing is done using Polars.
    """
    year = date.year
    month = date.month
    day = date.day
    pos_year = (month - 1) / 11.0
    days_in_month = calendar.monthrange(year, month)[1]
    pos_month = day / days_in_month
    dow = date.weekday()
    
    # Get meta info for the location from df_meta (Polars DataFrame)
    meta_row = df_meta.filter(pl.col("LocationID") == int(location_id)).row(0, named=True)
    lat = meta_row["LAT"]
    lon = meta_row["LONGTD"]
    
    # Normalize continuous features using cont_norm
    norm_year = (year - cont_norm["Year"][0]) / (cont_norm["Year"][1] - cont_norm["Year"][0])
    norm_lat = (lat - cont_norm["Lat"][0]) / (cont_norm["Lat"][1] - cont_norm["Lat"][0])
    norm_lon = (lon - cont_norm["Lon"][0]) / (cont_norm["Lon"][1] - cont_norm["Lon"][0])
    X_cont_arr = np.array([norm_year, pos_year, pos_month, norm_lat, norm_lon], dtype=np.float64)
    
    def one_hot(location_id, direction, county, hwy, dow):
        vec = np.zeros(len(dummy_vocab), dtype=np.float64)

        target = f"LocationID_{str(location_id)}"
        idx = dummy_vocab[target]
        vec[idx] = 1.0

        target = f"Direction_{str(direction)}"
        idx = dummy_vocab[target]
        vec[idx] = 1.0

        target = f"COUNTYNAME_{str(county)}"
        idx = dummy_vocab[target]
        vec[idx] = 1.0

        target = f"HWYNUMB_{str(hwy)}"
        idx = dummy_vocab[target]
        vec[idx] = 1.0

        target = f"DayOfWeek_{str(dow)}"
        idx = dummy_vocab[target]
        vec[idx] = 1.0
        return vec
    
    # For COUNTYNAME and HWYNUMB, look them up in df_meta.
    meta_row = df_meta.filter(pl.col("LocationID") == int(location_id)).row(0, named=True)
    county = meta_row["COUNTYNAME"]
    hwy = meta_row["HWYNUMB"]
    
    X_cat_arr = one_hot(location_id, direction, county, hwy, dow)
    
    # Convert to torch tensors
    X_cont_tensor = torch.tensor(X_cont_arr).unsqueeze(0)
    X_cat_tensor = torch.tensor(X_cat_arr).unsqueeze(0)
    
    return X_cont_tensor, X_cat_tensor

# =============================================================================
# 2. PyTorch Dataset and DataLoaders (using random_split)
# =============================================================================

class TrafficDataset(Dataset):
    def __init__(self, X_cont, X_cat, Y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float64)
        self.X_cat = torch.tensor(X_cat, dtype=torch.float64)  # now one-hot (float)
        self.Y = torch.tensor(Y, dtype=torch.float64)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X_cont[idx], self.X_cat[idx], self.Y[idx]

def build_dataloaders(X_cont, X_cat, Y, batch_size=32):
    dataset = TrafficDataset(X_cont, X_cat, Y)
    total = len(dataset)
    train_size = int(0.8 * total)
    test_size = total - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# =============================================================================
# 3. Model with One-Hot Based "Embedding" (Linear layer)
# =============================================================================

class TrafficPredictorOneHot(nn.Module):
    def __init__(self, cont_dim=5, cat_dim=30, embed_dim=10, hidden_dim=256, output_dim=24):
        """
        cat_dim: dimensionality of the one-hot encoded categorical vector (should be 30)
        embed_dim: output dimension of the manual embedding layer for categorical features.
        """
        super(TrafficPredictorOneHot, self).__init__()
        # Instead of nn.Embedding, we use a linear layer to map the one-hot vector.
        self.embed_linear = nn.Linear(cat_dim, embed_dim, bias=False)  # no activation
        total_input_dim = cont_dim + embed_dim
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, x_cont, x_cat):
        cat_emb = self.embed_linear(x_cat)
        x = torch.cat([x_cont, cat_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        #x = self.dropout(x)
        x = self.relu(self.fc5(x))
        #x = self.dropout(x)
        x = self.relu(self.fc6(x))
        #x = self.dropout(x)
        out = self.fc7(x)
        return out

# =============================================================================
# 4. Training and Evaluation Loops
# =============================================================================

def train(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for x_cont, x_cat, targets in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_cont, x_cat)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * x_cont.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_cont, x_cat, targets in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            outputs = model(x_cont, x_cat)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * x_cont.size(0)
    return running_loss / len(loader.dataset)

# =============================================================================
# 5. Main Training Function
# =============================================================================

def train_model():
    if os.path.exists(cont_feature_file) and os.path.exists(cat_feature_file) and os.path.exists(target_file) and os.path.exists(dummy_vocab_file) and os.path.exists(normalize_file):
        X_cont = np.load(cont_feature_file)
        X_cat = np.load(cat_feature_file)
        Y = np.load(target_file)
        with open(dummy_vocab_file, "rb") as f:
            vocab = pickle.load(f)
        with open(normalize_file, "rb") as f:
            cont_norm = pickle.load(f)
        print("Processed data loaded.")
    else:
        X_cont, X_cat, Y, vocab, cont_norm = preprocess()
    
    print(f"Continuous features shape: {X_cont.shape}")
    print(f"Categorical features shape: {X_cat.shape}")
    print(f"Targets shape: {Y.shape}")

    batch_size = 1024
    train_loader, test_loader = build_dataloaders(X_cont, X_cat, Y, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_dim = X_cat.shape[1]
    model = TrafficPredictorOneHot(cont_dim=5, cat_dim=cat_dim, embed_dim=10, output_dim=24)
    model.to(device=device, dtype=torch.float64)

    criterion = nn.MSELoss()
    max_lr = 0.005
    base_lr = 0.00005
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr)
    num_epochs = 1000000

    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        test_losses = checkpoint["test_loss"]
        train_losses = checkpoint["train_loss"]
        best_test_loss = checkpoint["best_test"]
        epoch = checkpoint["epoch"]
    else:
        best_test_loss = float("inf")
        train_losses, test_losses = [], []
        epoch = 0
    
    plt.ion()
    epochs_since_best = 0
    
    while epoch < num_epochs:
        train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4e} | Test Loss: {test_loss:.4e} | best: {best_test_loss:.4e} ({epochs_since_best} epochs ago)")

        if test_loss > 2. * best_test_loss or train_loss > 2. * best_test_loss or epochs_since_best > 100:
            checkpoint = torch.load(model_file)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            test_losses = checkpoint["test_loss"]
            train_losses = checkpoint["train_loss"]
            best_test_loss = checkpoint["best_test"]
            epoch = checkpoint["epoch"]
            model = model.to(device=device, dtype=torch.float64)

            if batch_size < 10000:
                batch_size = int(1.1 * batch_size)
                train_loader, test_loader = build_dataloaders(X_cont, X_cat, Y, batch_size=batch_size)
                print(f"Reloading best checkpoint and tweaking batchsize to {batch_size}")
            epochs_since_best = 0
            continue

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        epoch += 1
        x = np.arange(len(train_losses))
        plt.clf()
        #plt.plot(x[-200:], train_losses[-200:], label="train")
        #plt.plot(x[-200:], test_losses[-200:], label="validate")
        plt.plot(x, train_losses, label="train")
        plt.plot(x, test_losses, label="validate")
        plt.grid()
        plt.legend()
        plt.draw()    
        plt.pause(0.01)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_since_best = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_losses,
                'best_test': best_test_loss,
                'train_loss': train_losses
                }, model_file)
        else:
            epochs_since_best += 1
    
    print("Training complete. Best test loss: {:.4f}".format(best_test_loss))

if __name__ == "__main__":
    train_model()
'''
def test_model():
    X_cont = pd.read_csv("processed_features_cont.csv").values
    X_cat = np.load("processed_features_cat.npy")
    Y = np.load("processed_targets.npy")

    train_loader, test_loader = build_dataloaders(X_cont, X_cat, Y, batch_size=1)

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("cont_norm.pkl", "rb") as f:
        cont_norm = pickle.load(f)

    # Build vocabulary sizes for the embedding layers
    vocab_sizes = {
        'LocationID': len(vocab['LocationID']),
        'Direction': len(vocab['Direction']),
        'COUNTYNAME': len(vocab['COUNTYNAME']),
        'HWYNUMB': len(vocab['HWYNUMB']),
        'DayOfWeek': 7  # fixed
    }
    
    model = TrafficPredictorEmb(vocab_sizes=vocab_sizes)
    model.to(dtype=torch.float64)
    if os.path.exists(model_file):
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    model = model.to(torch.float64)
    model = model.to(device)
    #model.eval()
    model.train()

    criterion = nn.MSELoss()
    hours = np.arange(24)
    with torch.no_grad():
        for x_cont, x_cat, targets in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)

            model.train()
            for _ in range(50):
                predicted_traffic = model(x_cont, x_cat)
                predicted_traffic = predicted_traffic.detach().cpu().numpy().flatten()
                plt.plot(hours, predicted_traffic, color='C0', alpha=0.1)
            model.eval()
            predicted_traffic = model(x_cont, x_cat)
            loss = criterion(predicted_traffic, targets)
            predicted_traffic = predicted_traffic.detach().cpu().numpy().flatten()
            plt.plot(hours, predicted_traffic, alpha=1.0, color='C0', label=f'Predicted (no dropout) loss: {loss:.2e}')
            plt.plot(hours, targets.detach().cpu().numpy().flatten(), color='C0', linestyle='dashed', alpha=1.0, label='Truth')
            plt.legend()
            plt.show()
    def iterate_weekdays(year, month):
        """Iterates through each weekday (Monday-Friday) in a given month and year.

        Args:
            year (int): The year.
            month (int): The month (1-12).
        """
        cal = calendar.Calendar()
        for day in cal.itermonthdates(year, month):
            if day.month == month and day.weekday() < 5:  # Check if it's the correct month and a weekday (0-4)
                yield day

    # Example usage:
    year = 2023
    #month = 4
    direction = 'SB'
    location_id = 26024
    #date_str ='2024-06-11'
    df_traffic, df_meta = load_directional_hourly()
    df_traffic['Date'] = df_traffic['DateTime'].dt.date
    hours = np.arange(24)

    for month in range(5, 12):
        for i, weekday in enumerate(iterate_weekdays(year, month)):
            date_str = weekday.strftime("%Y-%m-%d")

            actual = df_traffic[
                (df_traffic['LocationID'] == location_id) &
                (df_traffic['Direction'] == direction) &
                (df_traffic['Date'] == weekday)
            ][['Hour', 'Volume']].copy()

            if len(actual) != 24:
                print(f'Length of actual data is bad: len(actual) = {len(actual)}')
                continue

            actual_sorted = actual.sort_values(by='Hour')
            actual_traffic = actual_sorted['Volume'].values

            #plt.plot(hours, actual_traffic, color=f'C{i}', linestyle='dashed', alpha=0.3)
            plt.plot(hours, actual_traffic, color=f'C0', linestyle='dashed', alpha=0.3)

            x_cont, x_cat = map_input_feature(location_id, direction, date_str, df_meta, cont_norm, vocab)
            model.train()
            for _ in range(100):
                predicted_traffic = model(x_cont, x_cat)
                predicted_traffic = predicted_traffic.detach().cpu().numpy().flatten()
                plt.plot(hours, predicted_traffic, color=f'C{i}', alpha=0.1)
            model.eval()
            predicted_traffic = model(x_cont, x_cat)
            predicted_traffic = predicted_traffic.detach().cpu().numpy().flatten()
            plt.plot(hours, predicted_traffic, color=f'C{i}', alpha=1.0)
            plt.show()
    '''

