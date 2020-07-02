from matplotlib import pyplot as plt
import folium

def plot_routes(data, n_sample):
    '''Return mapped plots of data.
       data: n * seq_len (numpy array)
    '''
    m = folium.Map(location=[np.mean(data[:,40:]), np.mean(data[:,:40])], zoom_start=20)
    for i in range(n_sample):
        idx = np.random.choice(range(len(data)))
        lat = data[idx,40:]
        long = data[idx,:40]
        route = list(map(lambda x, y: (x, y), lat, long))
        folium.PolyLine(route, weight=2.5, opacity=0.8).add_to(m)
        folium.Marker(route[0], icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(route[-1], icon=folium.Icon(color='red')).add_to(m)
    return m