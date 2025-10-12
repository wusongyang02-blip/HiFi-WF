import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pyautogui
import random 
import os
import threading

def tshark_thread(file_name, filt1, filt2):
    os.system(f'tshark -i 5 -w {file_name} -f "(host {filt1}) or (host {filt2})"') 

def visit_thread(url):
    os.system(f'chrome -incognito --start-maximized {url}')  # Use the traceless mode to clear the cache of each access to the same page

url_path = '.../website.csv'  # Input the file used to save the URLs
out_path = '.../traffic/'     # Save the traffic file you have obtained


count_url = 500
count_repeat = 10
count_batch = 10

count = 0

df_url = pd.read_csv(url_path, header=None)

# Loop through the URLs in batches
for C in range(0, int(count_url / count_batch)):

    # Repeat the process for the specified number of times
    for B in range(count_repeat):

        # Iterate through the URLs in the current batch
        for A in range(C * count_batch, C * count_batch + count_batch):
            url1 = df_url.iloc[A, 1]
            url2 = df_url.iloc[A, 3]
            url3 = df_url.iloc[A, 5]
            url4 = df_url.iloc[A, 7]
                
            web1 = int(df_url.iloc[A, 0])
            web2 = int(df_url.iloc[A, 2])
            web3 = int(df_url.iloc[A, 4])
            web4 = int(df_url.iloc[A, 6])
            
            count += 1
            print('\r                                                             ', end='')
            print('\rcapturing:', A, '-', B, end='  ')
            print(str(count) + '/' + str(count_url * count_repeat), end='  ')
            print(url1, url2, url3, url4,  end='  ')

            # Start traffic capture thread using tshark
            t1 = threading.Thread(target=tshark_thread, 
                                  args=(out_path + f"{web1}-{web2}-{web3}-{web4}-{B}.pcap", url1, url3))
            t1.start()
            
            # Delay a few seconds before visiting the first website
            time.sleep(3)

            # Start the thread to visit the first URL
            t2 = threading.Thread(target=visit_thread, 
                                  args=(url1,))
            t2.start()

            # Random delay [5, 9) seconds before visiting the second website
            time.sleep(random.uniform(5, 9))

            # Start the thread to visit the second URL
            t3 = threading.Thread(target=visit_thread, 
                                  args=(url2,))
            t3.start()
                 
            # Random delay [5, 9) seconds before visiting the third website
            time.sleep(random.uniform(5, 9))

            # Start the thread to visit the third URL
            t4 = threading.Thread(target=visit_thread, 
                                  args=(url3,))
            t4.start()
            
            # Random delay [5, 9) seconds before visiting the third website
            time.sleep(random.uniform(5, 9))

            # Start the thread to visit the forth URL
            t5 = threading.Thread(target=visit_thread, 
                                  args=(url4,))
            t5.start()

            # Wait for the pages to load
            time.sleep(random.uniform(7, 12))
            loading_time = 0
            while pyautogui.pixel(93, 63) == (71, 71, 71):  # Check if the page is still loading
                time.sleep(1)
                loading_time += 1
                if loading_time >= 10:
                    break

            # Take a screenshot after the pages have loaded
            time.sleep(2)
            pyautogui.screenshot(out_path + f"{web1}-{web2}-{web3}-{web4}-{B}.jpg")
            time.sleep(1)

            # Kill the tshark and Chrome processes to clean up
            os.system('taskkill /F /IM tshark.exe')
            os.system('taskkill /F /IM chrome.exe')

            time.sleep(3)
    
print('\ndone!')