# T.Alongi modified from H.Shaddox 2018-12-17 from 7D_download.py

# import packages
import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
from datetime import timedelta, date

#DOWNLOAD Albacore Data
#OBS network - 2D Deployed 2010-8-25 to 2011-9-16

#Specify Client, Network, start and end dates
client = Client("IRIS")
network = "2D"
start = date(2010, 8, 25)
end = date(2011, 9, 16)

#Create station directories if they don't alredy exist 
station_list_dir = [ "OBS30", "OBS31", "OBS32", "OBS33"]
os.chdir("/auto/home/talongi/Pvf/Wfdata_nobackup/2D")
for sta in station_list_dir:
	if not os.path.exists(sta):
		os.makedirs(sta)

        
os.chdir("/auto/home/talongi/Pvf")


#Write out stations in a format obspy can read 
station_list = "OBS30,OBS31,OBS32,OBS33"

#First save station information 
inventory = client.get_stations(starttime=UTCDateTime(start), endtime=UTCDateTime(end), 
                                network=network, 
                                station=station_list)
#                                level="response", 
#                                filename = 'Wfdata_nobackup/2D/2D_response', 
#                                format = "xml")
#inventory.plot() # plots station locations on a map

#%%

#Function to loop through times - will use below to download waveform data 
def daterange(start, end):
    for n in range(int ((end - start).days)):
        yield start + timedelta(n)


# working example of get_waveforms for this data
#st = client.get_waveforms(network='2D', station=station_list, location="*", channel="?H?", 
#                          starttime=UTCDateTime(date(2010,10,30)), 
#                          endtime=UTCDateTime(date(2010,11,2)))        
  
    
#Download waveforms one day at a time and save to folder 
for day in daterange(start, end):
    date_utc = UTCDateTime(day)
    st = client.get_waveforms(network=network, 
                              station=station_list, 
                              location="*", 
                              channel='*', 
                              starttime=date_utc, 
                              endtime=date_utc + 86400)
	
    for tr in st:
        sta = tr.stats.station
        chan = tr.stats.channel	
        net = tr.stats.network
        start = tr.stats.starttime
        jday = start.julday
        year = start.year
        tr.write("Wfdata_nobackup/%s/%s/%s.%s.%s.%s.%s" %(net, sta, sta, net, chan, year, jday), 
                 format = 'MSEED')
    st.clear()


#%% checking other stations on catalina island, only CIA may have been in operation during this time

st_lst = 'AVC,CTD,CIS,CIA,CIU'
inventory = client.get_stations(starttime=UTCDateTime(start), endtime=UTCDateTime(end), 
                                network='CI', 
                                station='CIU',
                                level="response", 
                                filename = 'Wfdata_nobackup/_response', 
                                format = "xml")