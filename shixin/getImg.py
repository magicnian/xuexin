import random
import time
import urllib.request

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36',
    'Host': 'zxgk.court.gov.cn'
}

for i in range(10):
    url = 'http://zxgk.court.gov.cn/shixin/captcha.do?captchaId=59a938bb0d6d41da9f689ca73e46c4d7&random=' + str(
        random.random())
    request = urllib.request.Request(url, headers=header)
    response = urllib.request.urlopen(request).read();
    with open('E:\\document\\shixin\\pic\\' + str(i) + '.jpg', 'wb') as f:
        f.write(response)
    time.sleep(1)
