# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:55:47 2020
The code is created to batchly download the data from Bureau of Reclamation for Yakima river basin under Yakima Project Hydromet System.
Realtime information: https://www.usbr.gov/pn/hydromet/yakima/yaktea.html
Data access protal: https://www.usbr.gov/pn/hydromet/yakima/yakwebarcread.html
Parameter code: https://www.usbr.gov/pn/hydromet/data/hydromet_pcodes.html
@author: CYLin
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os 
from tqdm import tqdm

output_path = r"C:\Users\cl2769\Documents\GitHub\YakimaRiverBasin\data\hydromet_raw"
stn_code_path = r"C:\Users\cl2769\Documents\GitHub\YakimaRiverBasin\data\BoR_Yakima_Stn_CodeKey.csv"

#%% Extract all data of Yakima river basin to txt
StnList = pd.read_csv(stn_code_path)["Code"].tolist()

for codekey in tqdm(StnList, desc = "Crawling process start"): 
    url = "https://www.usbr.gov/pn-bin/yak/webarccsv.pl?station="+ codekey+"&year=1895&month=1&day=1&year=2023&month=12&day=31&pcode=AF&pcode=MX&pcode=FB&pcode=MN&pcode=GD&pcode=MM&pcode=QD&pcode=QJ&pcode=QU&pcode=PP&pcode=WZ&pcode=WI&pcode=WK&pcode=PX&pcode=SP&pcode=PC&pcode=SY&pcode=PU&pcode=SO&pcode=TA&pcode=UA&pcode=UD&pcode=WR&pcode=WG&pcode=QT&pcode=GJ"
    html = requests.get(url, verify=False)
    soup = BeautifulSoup(html.text,'html.parser')
    soup_pretty = soup.prettify()
    data = soup.find_all("pre")[1].text
    
    # Save to txt files
    file = open(os.path.join(output_path, codekey + ".txt"),"w") 
    file.write(data)
    file.close()