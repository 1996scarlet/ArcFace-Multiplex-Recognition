import os
import time
import schedule
from multiprocessing import Process

# address_dict = ['10.41.0.208']
address_dict = ['10.41.0.208', '10.41.0.210', '10.41.0.211', '10.41.0.212']

cmd_string = "ps a | grep 'python3 peropero_v1.py {}'"
bootsetup = "gnome-terminal --tab --working-directory=$('pwd') --title={} -- python3 peropero_v1.py {}"
memcache_string = 'memcached -d -m 10 -u humanmotion -l 127.0.0.1 -p 12000 -c 256 -P /tmp/memcached.pid'

# sudo netstat -pl | grep memcached
# kill `cat /tmp/memcached.pid`
# sudo apt install xdotool

os.system(memcache_string)

Process(target=lambda: (
    os.system('python3 ../../cds_server.py'))
).start()


def job():
    for item in address_dict:

        res_string = os.popen(cmd_string.format(item)).readlines()

        if (len(res_string) <= 2):
            Process(target=lambda: (
                os.system(bootsetup.format(item, item)))
            ).start()


schedule.every(15).minutes.do(job)

job()

while True:
    schedule.run_pending()
    time.sleep(128)
