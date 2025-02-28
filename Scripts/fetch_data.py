# first import the functions for downloading data from NWIS
import dataretrieval.nwis as nwis
import time

# specify the USGS site code for which we want data.
site = '02408540'
now_time = time.time()
nw_lon=-87.7
nw_lat=34.39
se_lon=-83.9
se_lat=31.17

# get instantaneous values (iv)
# df = nwis.get_record(sites=site, service='iv', start='2017-12-31', end='2018-01-01')

# # get water quality samples (qwdata)
# df2 = nwis.get_record(sites=site, service='qwdata', start='2017-12-31', end='2018-01-01')

# # get basic info about the site
# df3 = nwis.get_record(sites=site, service='site')
df = nwis.get_iv(site=site, start=now_time - 86400, end=now_time, ssl_check=True)
print(df)

# df = nwis.get_record(sites=site, multi_index=True, parameterCd='00060',ssl_check=True)
# print(df)S