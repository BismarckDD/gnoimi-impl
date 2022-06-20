#!encoding=utf-8

import subprocess

yadiskLink = "https://yadi.sk/d/11eDCm7Dsn9GA"

base_file_num = 37
learn_file_num = 14

# download base files
for i in range(base_file_num):
    command = "curl " + "\"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" \
            + yadiskLink + "&path=/base/base_" + str(i).zfill(2) + "\""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    # print(str(out))
    wgetLink = str(out).split(',')[0][8:]
    wgetCommand = 'wget ' + wgetLink + ' -O base_' + str(i).zfill(2)
    print ("Downloading base chunk " + str(i).zfill(2) + " wgetCommand: " + wgetCommand)
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write(b'e')
    process.wait()

# download learn files
for i in range(learn_file_num):
    command = "curl " + "\"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" \
            + yadiskLink + "&path=/learn/learn_" + str(i).zfill(2) + "\""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = str(out).split(',')[0][8:]
    wgetCommand = 'wget ' + wgetLink + ' -O learn_' + str(i).zfill(2)
    print ("Downloading learn chunk " + str(i).zfill(2) + ' ...')
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write(b'e')
    process.wait()
