import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kasak/summer_pcp_ws/src/perception_pkg/install/perception_pkg'
