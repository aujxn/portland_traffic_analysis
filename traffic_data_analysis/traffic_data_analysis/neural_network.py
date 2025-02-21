import pandas as pd
import numpy as np
import os
import pickle
import calendar
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_directional_hourly
import torch
from torch import nn
#import traffic_data_analysis as tda

import logging
logger = logging.getLogger(__name__)

#df_traffic, df_meta = tda.utils.load_directional_hourly()
#feature_file = "processed_features.csv"
#target_file = "processed_targets.npy"
#model_file = "best_model_emb.pth"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_hourly_outliers(df, lower_quantile=0.02, upper_quantile=0.98):
    '''
    Remove extreme outliers in the 'Volume' column for each hour group in the DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'Hour' and 'Volume'.
    lower_quantile : float, optional
        Lower quantile threshold (default 0.05).
    upper_quantile : float, optional
        Upper quantile threshold (default 0.95).
    
    Returns
    -------
    df_filtered : pandas.DataFrame
        DataFrame with outliers removed from each hour group.
    '''
    def filter_group(group):
        lower = group['Volume'].quantile(lower_quantile)
        upper = group['Volume'].quantile(upper_quantile)
        return group[(group['Volume'] >= lower) & (group['Volume'] <= upper)]
    

    df_filtered = df.groupby('Hour', group_keys=False)[['Volume', 'Hour']].apply(filter_group)
    return df_filtered

# =============================================================================
# 1. Preprocessing Functions
# =============================================================================

def preprocess():
    """
    Processes raw data from traffic and meta tables into features suitable for
    a neural network with embedding inputs.
    
    Returns:
      X_cont: continuous features as a NumPy array (n_days, 5)
      X_cat: categorical indices as a NumPy array (n_days, 5)
      Y: targets as a NumPy array (n_days, 24)
      vocab: dictionary mapping categorical column names to index mappings
      cont_norm: dictionary with min/max values used for normalization of continuous features
    """
    df_traffic, df_meta = load_directional_hourly()

    '''
    # ---------------------------------------------------------------------
    # 1. Compute outlier bounds for each LocationID, Direction, and Hour
    # ---------------------------------------------------------------------
    low_bounds = (
        df_traffic.groupby(['LocationID', 'Direction', 'Hour'])['Volume']
        .quantile(0.03)
        .reset_index()
        .rename(columns={'Volume': 'low'})
    )
    high_bounds = (
        df_traffic.groupby(['LocationID', 'Direction', 'Hour'])['Volume']
        .quantile(0.97)
        .reset_index()
        .rename(columns={'Volume': 'high'})
    )
    bounds = pd.merge(low_bounds, high_bounds, on=['LocationID', 'Direction', 'Hour'])
    # ---------------------------------------------------------------------   
    '''

    df_traffic['Date'] = df_traffic['DateTime'].dt.date
    grouped = df_traffic.groupby(['LocationID', 'Direction', 'Date'])
    complete_days = grouped.filter(lambda x: len(x) == 24).copy()

    daily_list = []
    # Group by LocationID, Direction, and Date.
    for (loc, dir_, dt), group in complete_days.groupby(['LocationID', 'Direction', 'Date']):
        group_sorted = group.sort_values(by='Hour')
        
        '''
        # -----------------------------------------------------------
        # Outlier filtering: For each hour, ensure the volume is within the bounds.
        valid = True
        for _, r in group_sorted.iterrows():
            # Look up the corresponding bounds for this (loc, dir_, hour)
            b = bounds[
                (bounds['LocationID'] == loc) &
                (bounds['Direction'] == dir_) &
                (bounds['Hour'] == r['Hour'])
            ]
            if not b.empty:
                if (r['Volume'] < b.iloc[0]['low']) or (r['Volume'] > b.iloc[0]['high']):
                    valid = False
                    break
        if not valid:
            continue  # Skip this day if any hour is out-of-bounds.
        # -----------------------------------------------------------
        '''

        group_sorted = group.sort_values(by='Hour')
        Y = group_sorted['Volume'].values  # shape (24,)
        rec = group_sorted.iloc[0]
        year = rec['Year']
        month = rec['DateTime'].month  # month 1-12
        day = rec['DateTime'].day
        pos_year = (month - 1) / 11.0
        days_in_month = calendar.monthrange(year, month)[1]
        pos_month = day / days_in_month
        day_of_week = rec['DateTime'].weekday()  # 0 (Mon) ... 6 (Sun)
        meta_row = df_meta[df_meta['LocationID'] == loc].iloc[0]
        lat = meta_row['LAT']
        lon = meta_row['LONGTD']
        daily_list.append({
            'LocationID': loc,
            'Direction': dir_,
            'Date': dt,
            'Year': year,
            'Month': month,
            'Day': day,
            'PosYear': pos_year,
            'PosMonth': pos_month,
            'DayOfWeek': day_of_week,
            'Lat': lat,
            'Lon': lon,
            'Y': Y
        })

    daily_df = pd.DataFrame(daily_list)
    
    daily_df = daily_df.merge(df_meta[['LocationID', 'COUNTYNAME', 'HWYNUMB']], on='LocationID', how='left')
    
    # We'll create continuous features: Year, PosYear, PosMonth, Lat, Lon.
    cont_features = daily_df[['Year', 'PosYear', 'PosMonth', 'Lat', 'Lon']].copy()
    cont_min = cont_features.min()
    cont_max = cont_features.max()
    cont_norm = {col: (cont_min[col], cont_max[col]) for col in cont_features.columns}
    cont_features = (cont_features - cont_min) / (cont_max - cont_min)
    
    # Categorical features: 
    # We need indices for: LocationID, Direction, COUNTYNAME, HWYNUMB, DayOfWeek.
    # Convert to string if necessary.
    daily_df['LocationID'] = daily_df['LocationID'].astype(str)
    daily_df['Direction'] = daily_df['Direction'].astype(str)
    daily_df['COUNTYNAME'] = daily_df['COUNTYNAME'].astype(str)
    daily_df['HWYNUMB'] = daily_df['HWYNUMB'].astype(str)
    daily_df['DayOfWeek'] = daily_df['DayOfWeek'].astype(int)
    
    vocab = {}
    for col in ['LocationID', 'Direction', 'COUNTYNAME', 'HWYNUMB']:
        unique_vals = sorted(daily_df[col].unique())
        vocab[col] = {val: i for i, val in enumerate(unique_vals)}
    # For DayOfWeek, we assume fixed: 0-6
    vocab['DayOfWeek'] = {i: i for i in range(7)}
    
    # Create index columns
    daily_df['Location_idx'] = daily_df['LocationID'].map(vocab['LocationID'])
    daily_df['Direction_idx'] = daily_df['Direction'].map(vocab['Direction'])
    daily_df['County_idx'] = daily_df['COUNTYNAME'].map(vocab['COUNTYNAME'])
    daily_df['Highway_idx'] = daily_df['HWYNUMB'].map(vocab['HWYNUMB'])
    daily_df['DOW_idx'] = daily_df['DayOfWeek']  # already integer
    
    # Build feature arrays
    X_cont = cont_features.values  # shape (n_days, 5)
    X_cat = daily_df[['Location_idx', 'Direction_idx', 'County_idx', 'Highway_idx', 'DOW_idx']].values.astype(np.int64)  # (n_days, 5)
    Y = np.vstack(daily_df['Y'].values)  # shape (n_days, 24)
    
    print(f'Shape of target data: {Y.shape}')
    # Save processed dataset
    pd.DataFrame(X_cont, columns=cont_features.columns).to_csv("processed_features_cont.csv", index=False)
    np.save("processed_features_cat.npy", X_cat)
    np.save("processed_targets.npy", Y)
    
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("cont_norm.pkl", "wb") as f:
        pickle.dump(cont_norm, f)

    return X_cont, X_cat, Y, vocab, cont_norm

def map_input_feature(location_id, direction, date_str, df_meta, cont_norm, vocab):
    """
    Given a location_id, direction, and date string (yyyy-mm-dd),
    returns a tuple (X_cont, X_cat) corresponding to the processed input feature.
    Normalization parameters (cont_norm) and vocab dictionaries are used.
    """
    dt = pd.to_datetime(date_str)
    year = dt.year
    month = dt.month
    day = dt.day
    pos_year = (month - 1) / 11.0
    days_in_month = calendar.monthrange(year, month)[1]
    pos_month = day / days_in_month
    dow = dt.weekday()
    
    # Get meta information
    meta_row = df_meta[df_meta["LocationID"] == location_id].iloc[0]
    lat = meta_row["LAT"]
    lon = meta_row["LONGTD"]
    
    # Normalize continuous features using cont_norm (which contains (min, max) for each column)
    norm_year = (year - cont_norm["Year"][0]) / (cont_norm["Year"][1] - cont_norm["Year"][0])
    # pos_year and pos_month are already in [0,1]
    norm_lat = (lat - cont_norm["Lat"][0]) / (cont_norm["Lat"][1] - cont_norm["Lat"][0])
    norm_lon = (lon - cont_norm["Lon"][0]) / (cont_norm["Lon"][1] - cont_norm["Lon"][0])
    
    X_cont = np.array([norm_year, pos_year, pos_month, norm_lat, norm_lon])
    
    # For categorical features, convert location_id and meta fields to strings.
    loc_idx = vocab['LocationID'][str(location_id)]
    dir_idx = vocab['Direction'][direction]
    county_idx = vocab['COUNTYNAME'][str(meta_row["COUNTYNAME"])]
    hwy_idx = vocab['HWYNUMB'][str(meta_row["HWYNUMB"])]
    dow_idx = dow  # already integer
    
    X_cat = np.array([loc_idx, dir_idx, county_idx, hwy_idx, dow_idx])
    #return X_cont, X_cat
    return torch.tensor(X_cont, dtype=torch.float64).to(device), torch.tensor(X_cat, dtype=torch.long).to(device)
# =============================================================================
# 2. PyTorch Dataset and DataLoaders (using random_split)
# =============================================================================

class TrafficDataset(Dataset):
    def __init__(self, X_cont, X_cat, Y):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float64)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)  # indices for embeddings
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
# 3. Model with Embedding Layers
# =============================================================================

class TrafficPredictorEmb(nn.Module):
    def __init__(self, cont_dim=5, vocab_sizes=None, embed_dim=3, hidden_dim=1064, output_dim=24):
        """
        vocab_sizes: dictionary with keys: 'LocationID', 'Direction', 'COUNTYNAME', 'HWYNUMB', 'DayOfWeek'
                     Each value is the number of unique categories.
        """
        super(TrafficPredictorEmb, self).__init__()
        # Embedding layers for each categorical variable
        self.emb_location = nn.Embedding(vocab_sizes['LocationID'], 6)
        self.emb_direction = nn.Embedding(vocab_sizes['Direction'], embed_dim)
        self.emb_county = nn.Embedding(vocab_sizes['COUNTYNAME'], embed_dim)
        self.emb_hwy = nn.Embedding(vocab_sizes['HWYNUMB'], embed_dim)
        self.emb_dow = nn.Embedding(vocab_sizes['DayOfWeek'], embed_dim)  # fixed at 7
        
        total_emb_dim = (4 * embed_dim) + 6
        total_input_dim = cont_dim + total_emb_dim
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x_cont, x_cat):
        # x_cat: shape (batch, 5)
        x_cat = x_cat.reshape(-1, 5)
        x_cont = x_cont.reshape(-1, 5)
        emb_loc = self.emb_location(x_cat[:, 0])
        emb_dir = self.emb_direction(x_cat[:, 1])
        emb_county = self.emb_county(x_cat[:, 2])
        emb_hwy = self.emb_hwy(x_cat[:, 3])
        emb_dow = self.emb_dow(x_cat[:, 4])
        # Concatenate embeddings along dim=1
        emb = torch.cat([emb_loc, emb_dir, emb_county, emb_hwy, emb_dow], dim=1)
        x = torch.cat([x_cont, emb], dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        out = torch.special.expm1(x)
        return out

# =============================================================================
# 4. Training and Testing Loops
# =============================================================================

def train(model, loader, criterion, optimizer, device):
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
    # Load processed features if available, else run preprocess()
    if os.path.exists("processed_features_cont.csv") and os.path.exists("processed_features_cat.npy") and os.path.exists("processed_targets.npy") and os.path.exists('vocab.pkl') and os.path.exists('cont_norm.pkl'):
        X_cont = pd.read_csv("processed_features_cont.csv").values
        X_cat = np.load("processed_features_cat.npy")
        Y = np.load("processed_targets.npy")

        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open("cont_norm.pkl", "rb") as f:
            _cont_norm = pickle.load(f)
        print("Processed data loaded.")
    else:
        X_cont, X_cat, Y, vocab, _cont_norm = preprocess()
    
    batch_size = 64
    train_loader, test_loader = build_dataloaders(X_cont, X_cat, Y, batch_size=batch_size)
    
    # Build vocabulary sizes for the embedding layers
    vocab_sizes = {
        'LocationID': len(vocab['LocationID']),
        'Direction': len(vocab['Direction']),
        'COUNTYNAME': len(vocab['COUNTYNAME']),
        'HWYNUMB': len(vocab['HWYNUMB']),
        'DayOfWeek': 7  # fixed
    }
    
    model = TrafficPredictorEmb(vocab_sizes=vocab_sizes)
    if os.path.exists(model_file):
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    model = model.to(torch.float64)
    model = model.to(device)
    model.train()
    
    plt.ion()
    train_losses = []
    test_losses = []

    lr = 0.002
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_epochs = 1e100
    epoch = 0
    iters_since_best = 0
    best_test_loss = evaluate(model, test_loader, criterion, device)
    while epoch < num_epochs:
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | best: {best_test_loss:.4f}")

        if test_loss > 1.3 * best_test_loss or train_loss > 1.3 * best_test_loss or iters_since_best > 50:
            state_dict = torch.load(model_file)
            model.load_state_dict(state_dict)
            model = model.to(torch.float64)
            model = model.to(device)
            lr *= 0.9
            batch_size = int(1.1 * batch_size)
            if batch_size > 5000:
                batch_size = 5000
            if lr < 5e-4:
                lr = 5e-4
            print(f"Tweaking batchsize and learning rate to {batch_size} and {lr}")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_loader, test_loader = build_dataloaders(X_cont, X_cat, Y, batch_size=batch_size)
            iters_since_best = 0
            continue

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        epoch += 1
        x = np.arange(len(train_losses))
        plt.clf()
        plt.plot(x[-1000:], train_losses[-1000:], label="train")
        plt.plot(x[-1000:], test_losses[-1000:], label="validate")
        plt.grid()
        plt.legend()
        plt.draw()    
        plt.pause(0.001)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            iters_since_best = 0
            torch.save(model.state_dict(), model_file)
        else:
            iters_since_best += 1

    
    print("Training complete. Best test loss: {:.4f}".format(best_test_loss))

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
    '''
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
            


if __name__ == "__main__":
    #train_model()
    test_model()
