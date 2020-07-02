import re
import random
from ctgan import CTGANSynthesizer

def data_criteria(D, seq_len):
    '''Return a list of indices of samples that have assigned sequence length.
       D: pandas dataframe
       seq_len: int
    '''
    idx_lst = []
    for i in range(len(D)):
        gps_lst = re.sub(r"[[|[|]|]|]]", "", str(D["POLYLINE"][i])).split(",")
        if len(gps_lst) == seq_len*2:
            idx_lst.append(i)
    return idx_lst

def revise_data(it, D, cate, seq_len):
    '''Return tabulated GPS data along with rest of the features.
       it: number of samples needed. (int)
       D: already categorized by call type. (pandas dataframe)
       cate: call type. (str)
       seq_len: int
    '''
    col = ['duration','distance','avg_d_long', 'avg_d_lat', 's_long', 's_lat']
    M = pd.DataFrame(np.zeros((it, len(col))), columns=col)
    
    idx_lst = data_criteria(D, seq_len) # route steps
    idx = np.random.choice(idx_lst)
    gps_lst = re.sub(r"[[|[|]|]|]]", "", str(D["POLYLINE"][idx])).split(",")
    long = gps_lst[::2]
    lat = gps_lst[1::2]
    
    for i in range(it):
        M['duration'][i] = (len(long)-1)*15/60 # minutes
        M['s_long'][i] = float(long[0])
        M['s_lat'][i] = float(lat[0])
        dist, d_long, d_lat = 0, 0, 0
        for j in range(len(long)-1):
            dist += np.sqrt((float(long[j+1])-float(long[j]))**2 + (float(lat[j+1])-float(lat[j]))**2)
            d_long += float(long[j+1]) - float(long[j])
            d_lat += float(lat[j+1]) - float(lat[j])
    
        M['distance'][i] = dist
        M['avg_d_long'][i] = d_long/(len(long)-1)
        M['avg_d_lat'][i] = d_lat/(len(lat)-1)
    
    if cate == 'A':
        sub_D = D[['TRIP_ID','ORIGIN_CALL','TIMESTAMP','TAXI_ID','DAY_TYPE','MISSING_DATA']].iloc[:it,:]
    elif cate == 'B':
        sub_D = D[['TRIP_ID','ORIGIN_STAND','TIMESTAMP','TAXI_ID','DAY_TYPE','MISSING_DATA']].iloc[:it,:]
    else:
        sub_D = D[['TRIP_ID','TIMESTAMP','TAXI_ID','DAY_TYPE','MISSING_DATA']].iloc[:it,:]
    
    data = pd.concat([sub_D, M], axis=1)
    data = data.replace([np.nan, True, False], [0, 1, 0])
    return data, idx_lst

def gen_data(data, dis_col, repeat, n_sample):
    '''Return tabulated synthsized data.
       data: pandas dataframe
       dis_col: list of column names of discrete data. (list)
       repeat: training epochs. (int)
       n_sample: number of samples generated. (int)
    '''
    ctgan = CTGANSynthesizer()
    ctgan.fit(data, dis_col, epochs=repeat)
    syn = ctgan.sample(n_sample)
    return syn

def train_one_path(it, D, cate, repeat, n_sample, seq_len):
    '''Return tabulated synthesized data.'''
    Feature, idx_lst = revise_data(it, D, cate, seq_len)
    if cate == 'A':
        col_name = ['ORIGIN_CALL','TAXI_ID','DAY_TYPE','MISSING_DATA']
    elif cate == 'B':
        col_name = ['ORIGIN_STAND','TAXI_ID','DAY_TYPE','MISSING_DATA']
    else:
        col_name = ['TAXI_ID','DAY_TYPE','MISSING_DATA']
    
    syn = gen_data(Feature, col_name, repeat, n_sample)
    return syn, idx_lst

def gen_path(t_m, t_s, long_m, long_s, lat_m, lat_s, s_long_m, s_long_s, s_lat_m, s_lat_s, step):
    '''Return generated path from mean and std of tabulated GPS data.'''
    long, lat = [], []
    long.append(np.random.normal(s_long_m, s_long_s))
    lat.append(np.random.normal(s_lat_m, s_lat_s))
    for i in range(step):
        long.append(long[i]+np.random.normal(long_m, long_s))
        lat.append(lat[i]+np.random.normal(lat_m, lat_s))
    return long, lat

def plot_path(syn):
    '''Return generated path from tabulated GPS data'''
    long, lat = gen_path(
        np.mean(syn['duration']), np.std(syn['duration']),
        np.mean(syn['avg_d_long']), np.std(syn['avg_d_long']),
        np.mean(syn['avg_d_lat']), np.std(syn['avg_d_lat']),
        np.mean(syn['s_long']), np.std(syn['s_long']),
        np.mean(syn['s_lat']), np.std(syn['s_lat'])
    )
    return long+lat

def sample_real(D, lst, n_sample):
    '''Return randomly sampled real data with defined seq_len.'''
    M = []
    for i in range(n_sample):
        idx = np.random.choice(lst)
        gps_lst = re.sub(r"[[|[|]|]|]]", "", str(D["POLYLINE"][idx])).split(",")
        long = gps_lst[::2]
        lat = gps_lst[1::2]
        long = [float(x) for x in long]
        lat = [float(y) for y in lat]
        M.append(long+lat)
    df = pd.DataFrame(M)
    return df

def train_ctgan(it, D, cate, repeat, n_sample, seq_len, epochs):
    syn_df = []
    for i in range(epochs):
        syn, idx_lst = train_one_path(it, D, cate, repeat, n_sample, seq_len)
        syn_route = plot_path(syn) # long+lat
        syn_df.append(syn_route)
    
    syn_df = pd.DataFrame(syn_df)
    return syn_df, idx_lst