import GlassdoorScrap as gs
import pandas as pd
path1 = "D:\\Git1\\webscraping\\chromedriver_win32\\chromedriver.exe"
df = gs.get_jobs(keyword='Data Scientist', num_jobs=1000, verbose=False, path= path1, slp_time=15)
df.to_csv('gsdata.csv', index=False)