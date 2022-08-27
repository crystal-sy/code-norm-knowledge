# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:43:34 2021

@author: styra
"""

import os

logging_path = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'logging.conf'
project_path = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")) + os.sep

# github
github_name = "crystal-sy"
github_pwd = "XXX"
github_url = "https://github.com"
github_login = "https://github.com/login"
github_headers = {
    'Referer': 'https://github.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Host': 'github.com'
}
