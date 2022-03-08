#Imports
import tqdm
import glob
import numpy as np
import pandas as pd
import xarray as xr

import tensorflow as tf

from sklearn.model_selection import train_test_split

#%%
path = 'C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/Yearly Data/'

# Reading datasets
ctl = xr.open_dataset(path + 'CTL.nc') #'CTL_AOD550.nc'
#ctl_masked = xr.open_dataset(path + 'CTL_AOD550_MASKED.nc') #'CTL_AOD550_MASKED.nc'
nat = xr.open_dataset(path + '001.nc') #'NAT1_AOD550.nc'
spex_one = xr.open_dataset(path + 'SPEXone_Mask.nc')


#%% Functions
def create_input_images(df, input_label):
    '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    input_image = []
    if input_label == 'spex_3_day':
        for i in df.index:
            input_image.append(df[input_label][i].reshape(96, 192, 1))
    else:
        for i in df.index:
            input_image.append(df[input_label][i].reshape(96, 192, 1))
    return np.array(input_image)

def create_output(df, output_label):
    '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    output_image = []
    for i in df.index:
        output_image.append(df[output_label][i].reshape(96, 192, 1))
    return np.array(output_image)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_combined_data(ctl, ctl_lat, ctl_lon, spex, nat):

    # define the dictionary -- the structure -- of our single example
    data = {
        'height': _int64_feature(spex['aod_550'].shape[0]),
        'width': _int64_feature(spex['aod_550'].shape[1]),
        'depth': _int64_feature(spex['aod_550'].shape[2]),
    }
    
    for key in ctl.keys():
        data['ctl_'+key] = _bytes_feature(serialize_array(ctl[key]))
        data['ctl_lat_'+key] = _bytes_feature(serialize_array(ctl_lat[key]))
        data['ctl_lon_'+key] = _bytes_feature(serialize_array(ctl_lon[key]))
        data['spex_'+key] = _bytes_feature(serialize_array(spex[key]))
        data['nat_'+key] = _bytes_feature(serialize_array(nat[key]))

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_data(ctl, ctl_lat, ctl_lon, spex, nat,
               filename, max_files, out_dir):
    '''Writes the data to multiple tfrecord files each containing max_files examples'''
    splits = (len(spex['aod_550'])//max_files) + 1
    if len(spex['aod_550']) % max_files == 0:
        splits -= 1

    print(
        f"\nUsing {splits} shard(s) for {len(spex_one)} files,\
            with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}{}_{}.tfrecords".format(
            out_dir, i+1, splits, filename, max_files)
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i*max_files + current_shard_count
            if index == len(spex['aod_550']):
                break
            
            current_ctl = {}
            current_ctl_lat = {}
            current_ctl_lon = {}
            current_spex = {}
            current_nat = {}
            
            for key in ctl.keys():
                current_ctl[key] = ctl[key][index]
                current_ctl_lat[key] = ctl_lat[key][index]
                current_ctl_lon[key] = ctl_lon[key][index]
                current_spex[key] = spex[key][index]
                current_nat[key] = nat[key][index]

            out = parse_combined_data(ctl=current_ctl,                                    
                                      ctl_lat=current_ctl_lat,
                                      ctl_lon=current_ctl_lon,
                                      spex=current_spex,
                                      nat=current_nat)

            
            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count


def get_dataset_large(tfr_dir=path+'train_data/', 
                      pattern: str = "*train_data.tfrecords"):
    '''Loads the tfrecord files and returns a tfrecord dataset''' 
    files = glob.glob(tfr_dir+pattern, recursive=False)

    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(
        tf_parse)

    return dataset


def tf_parse(eg):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    example = tf.io.parse_example(
        eg[tf.newaxis],
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'day': tf.io.FixedLenFeature([], tf.int64),
            'spex_day': tf.io.FixedLenFeature([], tf.string),
            'ctl_lat_grad': tf.io.FixedLenFeature([], tf.string),
            'ctl_lon_grad': tf.io.FixedLenFeature([], tf.string),
            'ctl_day': tf.io.FixedLenFeature([], tf.string),
            'nat_day': tf.io.FixedLenFeature([], tf.string),
        },
    )
    spex_day = tf.io.parse_tensor(example["spex_day"][0], out_type="float32")
    ctl_lat_grad = tf.io.parse_tensor(example["ctl_lat_grad"][0], out_type="float32")
    ctl_lon_grad = tf.io.parse_tensor(example["ctl_lon_grad"][0], out_type="float32")
    ctl_day = tf.io.parse_tensor(example["ctl_day"][0], out_type="float32")
    nat_day = tf.io.parse_tensor(example["nat_day"][0], out_type="float32")
    input_data = tf.concat([spex_day, ctl_lat_grad, ctl_lon_grad, ctl_day], axis=-1)
    return input_data, nat_day

#%%

n_days = int(ctl.time.data.shape[0])
shape = (n_days, ctl.lat.shape[0], ctl.lon.shape[0])

CTL = {}
CTL_lat = {}
CTL_lon = {}
spex = {}
NAT = {}

parameters = ['aod_550', 'aod_865', 'aaod_550', 'ssa_550', 'ae']

for key in parameters:
    CTL[key] = {}
    CTL_lat[key] = {}
    CTL_lon[key] = {}
    spex[key] = {}
    NAT[key] = {}

for key in parameters:
    for n in range(n_days):
        if key == 'aod_550':
            CTL[key][n] = ctl.TAU_2D_550nm.data[n]
            NAT[key][n] = nat.TAU_2D_550nm.data[n]
            
        if key == 'aod_865':
            CTL[key][n] = ctl.TAU_2D_865nm.data[n]
            NAT[key][n] = nat.TAU_2D_865nm.data[n]
        
        if key == 'aaod_550':
            CTL[key][n] = ctl.ABS_2D_550nm.data[n]
            NAT[key][n] = nat.ABS_2D_550nm.data[n]
            
        if key == 'ssa_550':
            CTL[key][n] = ctl.OMEGA_2D_550nm.data[n]
            NAT[key][n] = nat.OMEGA_2D_550nm.data[n]
            
        if key == 'ae':
            CTL[key][n] = ctl.ANG_550nm_865nm.data[n]
            NAT[key][n] = nat.ANG_550nm_865nm.data[n]
            
        CTL_lat[key][n] = np.gradient(CTL[key][n], axis=0)
        CTL_lon[key][n] = np.gradient(CTL[key][n], axis=1)
        spex[key][n] = spex_one.Count.data[n]
        spex[key][n] = np.where(np.isnan(spex[key][n]), spex[key][n],
                                   NAT[key][n] * spex[key][n])
        spex[key][n] = np.nan_to_num(spex[key][n], nan=-1).reshape(96, 192, 1)

'''
spex_3_day = {}
for n in range(1, n_days-1):
    spex_3_day[n] = np.concatenate((spex_day_avg[n-1], spex_day_avg[n],
                                    spex_day_avg[n+1]), axis=-1)
    
#Remove first and last key value pair from original dictionary
del ctl_day_avg[0], ctl_day_avg[n_days-1]
del ctl_day_lat_grad[0], ctl_day_lat_grad[n_days-1]
del ctl_day_lon_grad[0], ctl_day_lon_grad[n_days-1]
del nat_day_avg[0], nat_day_avg[n_days-1]
'''

CTL_list, CTL_lat_list, CTL_lon_list, spex_list, NAT_list  = [], [], [], [], []
for key in parameters:
    CTL_list.append(pd.DataFrame(CTL[key].items(), columns=['Day', key]).drop('Day', axis=1))
    CTL_lat_list.append(pd.DataFrame(CTL_lat[key].items(), columns=['Day', key]).drop('Day', axis=1))
    CTL_lon_list.append(pd.DataFrame(CTL_lon[key].items(), columns=['Day', key]).drop('Day', axis=1))
    spex_list.append(pd.DataFrame(spex[key].items(), columns=['Day', key]).drop('Day', axis=1))
    NAT_list.append(pd.DataFrame(NAT[key].items(), columns=['Day', key]).drop('Day', axis=1))
    
CTL_df = pd.concat(CTL_list, axis=1)
CTL_lat_df = pd.concat(CTL_lat_list, axis=1)
CTL_lon_df = pd.concat(CTL_lon_list, axis=1)
spex_df = pd.concat(spex_list, axis=1)
NAT_df = pd.concat(NAT_list, axis=1)
    

#Input and output test data
ctl_aod_550_test = create_input_images(CTL_df, "aod_550")
ctl_aod_865_test = create_input_images(CTL_df, "aod_865")
ctl_aaod_550_test = create_input_images(CTL_df, "aaod_550")
ctl_ssa_550_test = create_input_images(CTL_df, "ssa_550")
ctl_ae_test = create_input_images(CTL_df, "ae")

ctl_lat_aod_550_test = create_input_images(CTL_lat_df, 'aod_550')
ctl_lat_aod_865_test = create_input_images(CTL_lat_df, 'aod_865')
ctl_lat_aaod_550_test = create_input_images(CTL_lat_df, 'aaod_550')
ctl_lat_ssa_550_test = create_input_images(CTL_lat_df, 'ssa_550')
ctl_lat_ae_test = create_input_images(CTL_lat_df, 'ae')

ctl_lon_aod_550_test = create_input_images(CTL_lon_df, 'aod_550')
ctl_lon_aod_865_test = create_input_images(CTL_lon_df, 'aod_865')
ctl_lon_aaod_550_test = create_input_images(CTL_lon_df, 'aaod_550')
ctl_lon_ssa_550_test = create_input_images(CTL_lon_df, 'ssa_550')
ctl_lon_ae_test = create_input_images(CTL_lon_df, 'ae')

spex_aod_550_test = create_input_images(spex_df, 'aod_550')
spex_aod_550_test = spex_aod_550_test.astype("float32")
spex_aod_865_test = create_input_images(spex_df, 'aod_865')
spex_aod_865_test = spex_aod_550_test.astype("float32")
spex_aaod_550_test = create_input_images(spex_df, 'aaod_550')
spex_aaod_550_test = spex_aod_550_test.astype("float32")
spex_ssa_550_test = create_input_images(spex_df, 'ssa_550')
spex_ssa_550_test = spex_aod_550_test.astype("float32")
spex_ae_test = create_input_images(spex_df, 'ae')
spex_ae_test = spex_aod_550_test.astype("float32")

nat_aod_550_test = create_output(NAT_df, "aod_550")
nat_aod_865_test = create_output(NAT_df, "aod_865")
nat_aaod_550_test = create_output(NAT_df, "aaod_550")
nat_ssa_550_test = create_output(NAT_df, "ssa_550")
nat_ae_test = create_output(NAT_df, "ae")

CTL = {'aod_550': ctl_aod_550_test, 'aod_865': ctl_aod_865_test, 'aaod_550': ctl_aaod_550_test,
       'ssa_550': ctl_ssa_550_test, 'ae': ctl_ae_test}
CTL_lat = {'aod_550': ctl_lat_aod_550_test, 'aod_865': ctl_lat_aod_865_test, 
           'aaod_550': ctl_lat_aaod_550_test,
           'ssa_550': ctl_lat_ssa_550_test, 'ae': ctl_lat_ae_test}
CTL_lon = {'aod_550': ctl_lon_aod_550_test, 'aod_865': ctl_lon_aod_865_test, 
           'aaod_550': ctl_lon_aaod_550_test,
           'ssa_550': ctl_lon_ssa_550_test, 'ae': ctl_lon_ae_test}
spex = {'aod_550': spex_aod_550_test, 'aod_865': spex_aod_865_test, 'aaod_550': spex_aaod_550_test,
        'ssa_550': spex_ssa_550_test, 'ae': spex_ae_test}
NAT = {'aod_550': nat_aod_550_test, 'aod_865': nat_aod_865_test, 'aaod_550': nat_aaod_550_test,
       'ssa_550': nat_ssa_550_test, 'ae': nat_ae_test}

#day_test = dataframe['Day'].values

#%% Write to tfrecord
write_data(CTL, CTL_lat, CTL_lon, spex, NAT, max_files=100,
           filename='train_multi_yearly', out_dir=path+"train_multi_yearly/")


#%% Testing
dataset = get_dataset_large()

for sample in dataset.take(1):
    print(repr(sample))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)

