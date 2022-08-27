# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:50:36 2021

@author: styra
"""

import random
import requests
import just
import lxml.html
from lxml import etree
from time import sleep

# sys
from config import cnk_config as cnk
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(cnk.logging_path)
logger = logging.getLogger('spider_data')

class Login(object):
    def __init__(self):
        self.headers = cnk.github_headers
        self.login_url = cnk.github_login
        self.post_url = 'https://github.com/session'
        self.logined_url = 'https://github.com/settings/profile'
        self.session = requests.Session()
    
    def token(self):
        response = self.session.get(self.login_url, headers=self.headers)
        selector = etree.HTML(response.text)
        token = selector.xpath('//div//input[2]/@value')
        return token
    
    def login(self, email, password):
        post_data = {
            'commit': 'Sign in',
            'utf8': '✓',
            'authenticity_token': self.token()[0],
            'login': email,
            'password': password
        }
        response = self.session.post(self.post_url, data=post_data, headers=self.headers)
        if response.status_code == 200:
            self.dynamics(response.text)
        
        response = self.session.get(self.logined_url, headers=self.headers)
        if response.status_code == 200:
            self.profile(response.text)
    
    def dynamics(self, html):
        selector = etree.HTML(html)
        dynamics = selector.xpath('//div[contains(@class, "news")]//div[contains(@class, "alert")]')
        for item in dynamics:
            dynamic = ' '.join(item.xpath('.//div[@class="title"]//text()')).strip()
            print(dynamic)
    
    def profile(self, html):
        selector = etree.HTML(html)
        name = selector.xpath('//input[@id="user_profile_name"]/@value')[0]
        email = selector.xpath('//select[@id="user_profile_email"]/option[@value!=""]/text()')
        print(name, email)


login = Login()
login.login(email=cnk.github_name, password=cnk.github_pwd)

# start collecting links
links = set()
query = "language:java filename:*.java"
for i in range(1, 2):
    try:
        url = "https://github.com/search?p={}&q={}&ref=searchresults&type=Code&utf8=%E2%9C%93"
        page_source = requests.get(url.format(i, query)).text
        tree = lxml.html.fromstring(page_source)
        page_links = [x for x in tree.xpath('//a/@href') if "/blob/" in x and "#" not in x]
        links.update(page_links)
        logger.info(u'正在下载代码 i:{i}, length:{l}'.format(i=i, l=len(links)))
        sleep(random.randint(6, 20))
    except KeyboardInterrupt:
        break

# visit and save source files
base = cnk.github_url
for num, link in enumerate(links):
    html = requests.get(base + link).text
    tree = lxml.html.fromstring(html)
    xpath = '//*[@class="blob-code blob-code-inner js-file-line"]'
    contents = "\n".join([x.text_content() for x in tree.xpath(xpath)])
    # note that link conveniently starts with / like a webpath
    just.write(contents, "data" + link)
    logger.info(u'下载代码内容 i:{i}, length:{l}'.format(i=num, l=len(contents)))
    sleep(random.randint(6, 20))
