import requests
import pandas as pd
import numpy


proxies = {'http': 'http://webproxy.bfm.com:8080'}

response = requests.get("http://www.google.com/finance/getprices?i=60&f=d,o,h,l,c,v&df=cpct&q=IBM", proxies=proxies)

response.raise_for_status()

with open('prices', 'wb') as handle:
    for block in response.iter_content(1024):
        handle.write(block)

prices = pd.read_csv('prices', sep=" ", delimiter= ",",header = None, skiprows= 7)

maximum = pd.DataFrame.max(prices)
minimum = pd.DataFrame.min(prices)

max_min = numpy.arange(maximum[2],minimum[1], -0.01)

print(type(max_min))



