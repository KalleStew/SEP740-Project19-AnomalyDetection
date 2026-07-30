# KDD Cup 1999 Data Dictionary

## Dataset Summary
- Source: KDD Cup 1999 intrusion detection benchmark
- Task: classify network connections as normal or attack traffic
- Record format: 41 input features plus one label column
- Label values: `normal.` or one of the attack names in the dataset
- Canonical schema source: `data/raw/kddcup.names`
- Attack family source: `data/raw/training_attack_types`

## Notebook Integration
- The EDA notebook loads feature names and declared feature types directly from `data/raw/kddcup.names` instead of hardcoding the schema.
- The notebook derives the symbolic feature list from that same metadata file, which keeps categorical analysis aligned with the source dataset definition.
- The attack-family grouping in the EDA notebook is loaded from `data/raw/training_attack_types` so the DoS, Probe, R2L, and U2R mapping is reproducible.

## Feature Groups

### Basic connection features
- `duration` - length of the connection in seconds
- `protocol_type` - network protocol, such as TCP, UDP, or ICMP
- `service` - destination network service
- `flag` - normal or error status of the connection
- `src_bytes` - bytes sent from source to destination
- `dst_bytes` - bytes sent from destination to source
- `land` - 1 if source and destination IP/port are identical, otherwise 0
- `wrong_fragment` - number of wrong fragments
- `urgent` - number of urgent packets

### Content features
- `hot` - number of hot indicators
- `num_failed_logins` - failed login attempts
- `logged_in` - 1 if successfully logged in, otherwise 0
- `num_compromised` - number of compromised conditions
- `root_shell` - 1 if a root shell is obtained, otherwise 0
- `su_attempted` - 1 if `su root` command attempted
- `num_root` - number of root accesses
- `num_file_creations` - number of file creation operations
- `num_shells` - number of shell prompts
- `num_access_files` - number of file access operations
- `num_outbound_cmds` - number of outbound commands; typically always 0 in this subset
- `is_host_login` - 1 if the login belongs to the host list, otherwise 0
- `is_guest_login` - 1 if the login is a guest login, otherwise 0

### Time-based traffic features
- `count` - number of connections to the same host in the past two seconds
- `srv_count` - number of connections to the same service in the past two seconds
- `serror_rate` - percentage of connections with SYN errors
- `srv_serror_rate` - SYN error rate for the same service
- `rerror_rate` - percentage of connections with REJ errors
- `srv_rerror_rate` - REJ error rate for the same service
- `same_srv_rate` - percentage of connections to the same service
- `diff_srv_rate` - percentage of connections to different services
- `srv_diff_host_rate` - percentage of connections to different hosts for the same service

### Host-based traffic features
- `dst_host_count` - number of connections to the same host
- `dst_host_srv_count` - number of connections to the same service
- `dst_host_same_srv_rate` - percentage of connections to the same service
- `dst_host_diff_srv_rate` - percentage of connections to different services
- `dst_host_same_src_port_rate` - percentage of connections from the same source port
- `dst_host_srv_diff_host_rate` - percentage of connections to the same service on different hosts
- `dst_host_serror_rate` - SYN error rate on the destination host
- `dst_host_srv_serror_rate` - SYN error rate for the same service on the destination host
- `dst_host_rerror_rate` - REJ error rate on the destination host
- `dst_host_srv_rerror_rate` - REJ error rate for the same service on the destination host
