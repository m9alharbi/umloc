import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pdb

def align_data(phone_data, position_data, start_index, end_index, sampling_rate, phone_cols, postion_cols):
    sensors_timestamp = phone_data['sensors_timestamp'].values
    position_timestamp = position_data['position_timestamp'].values
    start = sensors_timestamp[start_index]
    end = sensors_timestamp[-end_index]
    duration = (end - start)/1000
    num_samples = int(duration * sampling_rate)
    time_axis = np.linspace(start, end, num_samples)
    phone_interp_func = interp1d(sensors_timestamp, phone_data.values.T, kind='linear', fill_value="extrapolate")
    position_interp_func = interp1d(position_timestamp, position_data.values.T, kind='linear', fill_value="extrapolate")
    interp_phone_data = phone_interp_func(time_axis)
    interp_position_data = position_interp_func(time_axis)
    interp_phone_data = pd.DataFrame(interp_phone_data.T, columns=phone_cols)
    interp_position_data = pd.DataFrame(interp_position_data.T, columns=position_cols)
    interp_phone_data['time_stamp'] = time_axis
    data_df = pd.concat([interp_phone_data, interp_position_data], axis=1, join='inner')
    return data_df


position_data = pd.read_csv('/ibex/user/alham0a/localization_project/data/datasets/KAUST_dataset/250502_232911_ros_position_data.csv')
phone_data = pd.read_csv('/ibex/user/alham0a/localization_project/data/datasets/KAUST_dataset/raw_data250502_232926.csv')
phone_cols = phone_data.columns
position_cols = position_data.columns
plt.plot(phone_data['acceleration_x'].values)
plt.plot(position_data['position_x'].values)
plt.show()
start_index = 1000
end_index = 1000
sampling_rate = 30
position_data = position_data.sort_values('position_timestamp')
phone_data = phone_data.sort_values('sensors_timestamp')
df = align_data(phone_data, position_data, start_index, end_index, sampling_rate, phone_cols, position_cols)

df.to_csv('/home/mohammed/inertial_localization/dataset/s000_5.csv')


#s0002(230910_115130), s0003(231029_003357) s0005(231029_021737) start=1000 end=1000
#s0004(231029_012627) start=6000 end=2000
