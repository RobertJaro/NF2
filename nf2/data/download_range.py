import os
import sys

import drms
from dateutil.parser import parse

from nf2.data.download import donwload_ds

download_dir = sys.argv[1]
email = sys.argv[2]
harpnum = int(sys.argv[3])
t_start = parse(sys.argv[4])

os.makedirs(download_dir, exist_ok=True)
client = drms.Client(email=email, verbose=True)
ds = 'hmi.sharp_cea_720s[%d][%s/5d]{Br, Bp, Bt, Br_err, Bp_err, Bt_err}' % \
     (harpnum, t_start.isoformat('_', timespec='seconds'))
donwload_ds(ds, download_dir, client)