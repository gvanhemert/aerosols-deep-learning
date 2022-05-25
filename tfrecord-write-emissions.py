#Imports
import tqdm
import glob
import numpy as np
import pandas as pd
import xarray as xr
import pathlib

import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split

#%%
path = 'C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/Yearly Data/'
ensemble_path = Path('C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/Ensembles/')
emission_path = 'C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/Emissions/'
emission_path_2 = Path('C:/Users/Guus van Hemert/Desktop/TW/TW5/Thesis/Data/Emissions/')

ensembles = {}
ensembles['nat'] = {}
ensembles['ctl'] = {}
for a in emission_path_2.glob("*"):
    try:
        if str(a.parts[-1][-2:]) == 'nc' and int(a.parts[-1][1:3]) <= 8 and str(a.parts[-1][0]) == 'C':
            ensembles['ctl'][a.parts[-1][:-3]] = xr.open_dataset(a)
        if str(a.parts[-1][-2:]) == 'nc' and int(a.parts[-1][1:3]) <= 8 and str(a.parts[-1][0]) == 'N':
            ensembles['nat'][a.parts[-1][:-3]] = xr.open_dataset(a)
    except:
        print('Yikes')
        #if str(a.parts[-1][-2:]) == 'nc':
            #ensembles[a.parts[-1][:-3]] = xr.open_dataset(a)
    

# Reading datasets
ctl = xr.open_dataset(path + 'C01.nc')
nat = xr.open_dataset(path + 'NAT.nc')

ctl_emi = xr.open_dataset(emission_path + 'C01_emi.nc')
nat_emi = xr.open_dataset(emission_path + 'NAT_emi.nc')

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

def create_spex3_input_images(df, input_label):
    '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    input_image = []
    for i in df.index:
        input_image.append(df[input_label][i].reshape(96, 192, 1))
    return np.array(input_image)

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
    
    for key in nat.keys():
        data['nat_'+key] = _bytes_feature(serialize_array(nat[key]))
        data['ctl_'+key] = _bytes_feature(serialize_array(ctl[key]))
        if key in spex.keys():
            data['spex_'+key] = _bytes_feature(serialize_array(spex[key]))
            data['ctl_lat_'+key] = _bytes_feature(serialize_array(ctl_lat[key]))
            data['ctl_lon_'+key] = _bytes_feature(serialize_array(ctl_lon[key]))
        
        #if key in nat.keys():
            #data['nat_'+key] = _bytes_feature(serialize_array(nat[key]))

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
            
            for key in spex.keys():
                current_ctl[key] = ctl[key][index]
                current_ctl_lat[key] = ctl_lat[key][index]
                current_ctl_lon[key] = ctl_lon[key][index]
                current_spex[key] = spex[key][index]
                #current_nat[key] = nat[key][index]
            for key in nat.keys():
                current_ctl[key] = ctl[key][index]
                #current_ctl_lat[key] = ctl_lat[key][index]
                #current_ctl_lon[key] = ctl_lon[key][index]
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
spex_3_day = {}

parameters_1 = ['aod_550', 'ssa_550', 'ae']
parameters_2 = []

for key in parameters_1:
    CTL[key] = {}
    CTL_lat[key] = {}
    CTL_lon[key] = {}
    spex[key] = {}
    NAT[key] = {}
    spex_3_day[key] = {}

for ens_key in range(1):
    #ctl = ensembles['ctl']['C0'+str(ens_key+1)]
    #nat = ensembles['nat']['N0'+str(ens_key+1)]
    #ctl_emi = ensembles['ctl']['C0'+str(ens_key+1)+'_emi']
    #nat_emi = ensembles['nat']['N0'+str(ens_key+1)+'_emi']
    
    CTL = {}
    CTL_lat = {}
    CTL_lon = {}
    spex = {}
    NAT = {}
    spex_3_day = {}
    
    parameters_1 = ['aod_550', 'ssa_550', 'ae']
    parameters_2 = []

    for key in parameters_1:
        CTL[key] = {}
        CTL_lat[key] = {}
        CTL_lon[key] = {}
        spex[key] = {}
        NAT[key] = {}
        spex_3_day[key] = {}
    for key in parameters_1:
        for n in range(n_days):
            if key == 'aod_550':
                CTL[key][n] = ctl.TAU_2D_550nm.data[n]
                NAT[key][n] = nat.TAU_2D_550nm.data[n]
                
            if key == 'ssa_550':
                CTL[key][n] = ctl.OMEGA_2D_550nm.data[n]
                NAT[key][n] = nat.OMEGA_2D_550nm.data[n]
                
            if key == 'ae':
                CTL[key][n] = ctl.ANG_550nm_865nm.data[n]
                NAT[key][n] = nat.ANG_550nm_865nm.data[n]
            
            NAT_norm = (NAT[key][n] - np.min(NAT[key][n])) / (np.max(NAT[key][n]) - np.min(NAT[key][n]))
                
            CTL_lat[key][n] = np.gradient(CTL[key][n], axis=0)
            CTL_lon[key][n] = np.gradient(CTL[key][n], axis=1)
            spex[key][n] = spex_one.Count.data[n]
            spex[key][n] = np.where(np.isnan(spex[key][n]), spex[key][n],
                                      NAT[key][n] * spex[key][n])
            #spex[key][n] = np.nan_to_num(spex[key][n], nan=-1).reshape(96, 192, 1)
        for n in range(n_days-6):
            #spex_3_day[key][n] = np.concatenate([spex[key][n + i] for i in range(7)], axis=-1)
            spex_3_day[key][n] = np.nanmean(np.array([spex[key][n + 1] for i in range(7)]), axis=0)
            #spex_3_day[key][n] = np.nan_to_num(spex[key][n], nan=-1).reshape(96, 192, 1)
            CTL[key][n] = np.mean(np.array([CTL[key][n + i] for i in range(7)]), axis=0)
            NAT[key][n] = np.mean(np.array([NAT[key][n + i] for i in range(7)]), axis=0)
        for j in range(6):
            del CTL[key][n_days - j - 1]
            del CTL_lat[key][n_days - j - 1]
            del CTL_lon[key][n_days - j - 1]
            del NAT[key][n_days - j - 1]
    
    for name, var in nat_emi.items():
        NAT[name] = {}
        if name[0] == 'e':
            parameters_2.append(name)
            for n in range(n_days):
                NAT[name][n] = var.data[n]
            for n in range(n_days-6):
                NAT[name][n] = np.mean(np.array([NAT[name][n + i] for i in range(7)]), axis=0)
            for j in range(6):
                del NAT[name][n_days - j - 1]
    
    for name, var in ctl_emi.items():
        CTL[name] = {}
        #CTL_lat[name] = {}
        #CTL_lon[name] = {}
        if name[0] == 'e':
            for n in range(n_days):
                CTL[name][n] = var.data[n]
                #CTL_lat[name][n] = np.gradient(CTL[name][n], axis=0)
                #CTL_lon[name][n] = np.gradient(CTL[name][n], axis=1)
            for n in range(n_days-6):
                CTL[name][n] = np.mean(np.array([CTL[name][n + i] for i in range(7)]), axis=0)
                #CTL_lat[name][n] = np.gradient(CTL[name][n], axis=0)
                #CTL_lon[name][n] = np.gradient(CTL[name][n], axis=1)
                
            for j in range(6):
                del CTL[name][n_days-j-1]
                #del CTL_lat[name][n_days-j-1]
                #del CTL_lon[name][n_days-j-1]
    
    
    
    CTL_list, CTL_lat_list, CTL_lon_list, spex_list, NAT_list  = [], [], [], [], []
    for key in parameters_1:
        CTL_list.append(pd.DataFrame(CTL[key].items(), columns=['Day', key]).drop('Day', axis=1))
        CTL_lat_list.append(pd.DataFrame(CTL_lat[key].items(), columns=['Day', key]).drop('Day', axis=1))
        CTL_lon_list.append(pd.DataFrame(CTL_lon[key].items(), columns=['Day', key]).drop('Day', axis=1))
        spex_list.append(pd.DataFrame(spex_3_day[key].items(), columns=['Day', key]).drop('Day', axis=1))
        NAT_list.append(pd.DataFrame(NAT[key].items(), columns=['Day', key]).drop('Day', axis=1))
        
    for key in parameters_2:
        CTL_list.append(pd.DataFrame(CTL[key].items(), columns=['Day', key]).drop('Day', axis=1))
        #CTL_lat_list.append(pd.DataFrame(CTL_lat[key].items(), columns=['Day', key]).drop('Day', axis=1))
        #CTL_lon_list.append(pd.DataFrame(CTL_lon[key].items(), columns=['Day', key]).drop('Day', axis=1))
        NAT_list.append(pd.DataFrame(NAT[key].items(), columns=['Day', key]).drop('Day', axis=1))
        
    CTL_df = pd.concat(CTL_list, axis=1)
    CTL_lat_df = pd.concat(CTL_lat_list, axis=1)
    CTL_lon_df = pd.concat(CTL_lon_list, axis=1)
    spex_df = pd.concat(spex_list, axis=1)
    NAT_df = pd.concat(NAT_list, axis=1)
    
    CTL = {}
    CTL_lat = {}
    CTL_lon = {}
    spex = {}
    NAT = {}
    spex_3_day = {}
    
    #Input and output data
    for key in parameters_1:
        CTL[key] = create_input_images(CTL_df, key)
        CTL_lat[key] = create_input_images(CTL_lat_df, key)
        CTL_lon[key] = create_input_images(CTL_lon_df, key)
        spex[key] = create_spex3_input_images(spex_df, key)
        spex[key] = spex[key].astype("float32")
        NAT[key] = create_input_images(NAT_df, key)
        
    for key in parameters_2:
        CTL[key] = create_input_images(CTL_df, key)
        #CTL_lat[key] = create_input_images(CTL_lat_df, key)
        #CTL_lon[key] = create_input_images(CTL_lon_df, key)
        NAT[key] = create_input_images(NAT_df, key)
    
    write_data(CTL, CTL_lat, CTL_lon, spex, NAT, max_files=100,
               filename=str(ens_key+1), out_dir=emission_path+"Network/test_mean_emissions_7days_C01/")

#%% Write to tfrecord
write_data(CTL, CTL_lat, CTL_lon, spex, NAT, max_files=100,
           filename='test_mean_emissions_7days', out_dir=emission_path+"Network/test_mean_emissions_7days/")


#%% Testing
dataset = get_dataset_large()

for sample in dataset.take(1):
    print(repr(sample))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)

