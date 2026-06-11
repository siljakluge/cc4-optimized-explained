import subprocess
import numpy as np

# === Environment Structure Constants ===
SUBNETS = [
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet"
]

AGENT_SUBNETS = [
    "admin_network_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet"
]

HOSTS = [
    "router",
    "server_host_0",
    "server_host_1",
    "server_host_2",
    "server_host_3",
    "server_host_4",
    "server_host_5",
    "user_host_0",
    "user_host_1",
    "user_host_2",
    "user_host_3",
    "user_host_4",
    "user_host_5",
    "user_host_6",
    "user_host_7",
    "user_host_8",
    "user_host_9"
]

AGENTS = [f"blue_agent_{i}" for i in range(5)]

# === Communication Policy ===
# Maps mission phase to agent-specific subnet block policies
POLICY = {
    0: {
        "blue_agent_0": np.array([0,0,0,0,0,1,0,0,0]),
        "blue_agent_1": np.array([0,1,1,0,0,1,1,0,1]),
        "blue_agent_2": np.array([0,0,0,0,1,0,0,0,0]),
        "blue_agent_3": np.array([0,1,1,0,1,0,1,1,0]),
        "blue_agent_4": np.array([0,0,0,0,1,1,0,0,0])
    },
    0.5: {
        "blue_agent_0": np.array([0,1,1,0,1,1,1,0,1]),
        "blue_agent_1": np.array([0,1,1,0,0,1,1,1,1]),
        "blue_agent_2": np.array([0,0,0,0,1,0,0,1,0]),
        "blue_agent_3": np.array([0,1,1,0,1,0,1,1,0]),
        "blue_agent_4": np.array([0,0,0,0,1,1,0,0,0])
    },
    1: {
        "blue_agent_0": np.array([0,0,0,0,0,1,0,0,1]),
        "blue_agent_1": np.array([0,1,1,0,0,1,1,0,1]),
        "blue_agent_2": np.array([0,1,1,0,1,1,0,1,0]),
        "blue_agent_3": np.array([0,1,1,0,1,0,1,1,1]),
        "blue_agent_4": np.array([0,0,0,0,1,1,0,0,0])
    }
}


def get_agent_from_subnet(subnet):
    if subnet == "restricted_zone_a_subnet":
        return "blue_agent_0", 0
    elif subnet == "operational_zone_a_subnet":
        return "blue_agent_1" , 1
    elif subnet == "restricted_zone_b_subnet":
        return "blue_agent_2", 2
    elif subnet == "operational_zone_b_subnet":
        return "blue_agent_3", 3
    elif subnet == "office_network_subnet" or subnet == "admin_network_subnet" or subnet == "public_access_zone_subnet":
        return "blue_agent_4", 4
    else:
        raise ValueError(f"Unknown subnet: {subnet}")
    
def get_host_from_target(target):
    host = target.split("subnet_")[-1]
    subnet = target.split("_subnet")[0] + "_subnet"
    return host, subnet


def git_push(commit_message="auto commit"):
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Changes pushed successfully")
    except subprocess.CalledProcessError as e:
        print("❌ Git command failed:", e)

class heuristic_lambda_schedule:
    def __init__(self, initial_lambda=0.4, final_lambda=0.9, total_episodes=10000):
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.total_episodes = total_episodes
        self.episode = 0

    def get_lambda(self, rising = True):
        if not rising:
            if self.episode >= self.total_episodes:
                return self.initial_lambda
            else:
                # decrease lambda from final_lambda to initial_lambda over total_episodes
                return self.final_lambda + (self.initial_lambda - self.final_lambda) * (self.episode / self.total_episodes)
        else:    
            if self.episode >= self.total_episodes:
                return self.final_lambda
            else:
                return self.initial_lambda + (self.final_lambda - self.initial_lambda) * (self.episode / self.total_episodes)
    
    
def get_host_from_hostname(hostname):
    """
    Extract the host name from a hostname string.
    Args:
        hostname: Hostname string.
    Returns:
        Host name as string.
    """
    if "router" in hostname:
        host = "_".join(hostname.split("_")[-1:])
    else:
        host = "_".join(hostname.split("_")[-3:])
    return host

def update_to_block(lst, hostname, delete):
    """
    Update a list (queue) for blocking a given subnet, setting its bit.
    Args:
        lst: Numpy array representing the queue.
        hostname: Subnet name to update in the list.
        delete: If True, clear the bit; if False, set the bit.
    """
    if delete:
        value = 0
    else:
        value = 1
    index = SUBNETS.index(hostname)
    lst[index] = value
    return lst

def get_subnet_from_hostname(hostname):
    """
    Extract the subnet name from a hostname string.
    Args:
        hostname: Hostname string.
    Returns:
        Subnet name as string.
    """
    if "router" in hostname:
        subnet = "_".join(hostname.split("_")[:-1])
    else:
        subnet = "_".join(hostname.split("_")[:-3])
    return subnet

def get_agent_from_hostname(hostname):
    """
    Determine the agent responsible for a given hostname based on subnet.
    Args:
        hostname: Hostname string.
    Returns:
        Agent name as string.
    """
    subnet = get_subnet_from_hostname(hostname)
    if subnet == "restricted_zone_a_subnet":
        agent = "blue_agent_0"
    elif subnet == "operational_zone_a_subnet":
        agent = "blue_agent_1"
    elif subnet == "restricted_zone_b_subnet":
        agent = "blue_agent_2"
    elif subnet == "operational_zone_b_subnet":
        agent = "blue_agent_3"
    elif subnet == "admin_network_subnet" or subnet == "office_network_subnet" or subnet == "public_access_zone_subnet":
        agent = "blue_agent_4"
    else:
        raise AssertionError(f"Subnet not recognized: {subnet}")
    return agent

def transform_to_host(arr, prefix=""):
    """
    Transform a binary array into a list of hostnames, optionally with a prefix.
    Args:
        arr: Numpy array representing host selection.
        prefix: String or list of prefixes to prepend to hostnames.
    Returns:
        List of hostnames.
    """
    if isinstance(prefix, list):
        output = []
        hosts=[]
        for pre in prefix:
            hosts.extend([pre+ val for val in HOSTS])
            
        output += [val for i, val in enumerate(hosts) if arr[i] == 1]
        return output
    return [prefix+val for i, val in enumerate(HOSTS) if arr[i] == 1]

def transform_to_subnet(arr):
    """
    Transform a binary array into a list of subnet names.
    Args:
        arr: Numpy array representing subnet selection.
    Returns:
        List of subnet names.
    """
    return [val for i, val in enumerate(SUBNETS) if arr[i] == 1]

def get_subnet_from_agent(agent):
    """
    Get the subnet(s) associated with a given agent.
    Args:
        agent: Agent name string.
    Returns:
        Subnet name(s) as string or list of strings.
    """
    match agent:
        case "blue_agent_0":
            return "restricted_zone_a_subnet_"
        case "blue_agent_1":
            return "operational_zone_a_subnet_"
        case "blue_agent_2":
            return "restricted_zone_b_subnet_"
        case "blue_agent_3":
            return "operational_zone_b_subnet_"
        case "blue_agent_4":
            return ["admin_network_subnet_", "office_network_subnet_", "public_access_zone_subnet_"]
        case _:
            raise AssertionError(f"Agent {agent} not recognized")
        
def update_vector(vec, hostname, delete=True):
    """
    Update a vector (queue) for a given hostname, setting or clearing its bit.
    Args:
        vec: Numpy array representing the queue.
        hostname: Hostname to update in the vector.
        delete: If True, clear the bit; if False, set the bit.
    Returns:
        Updated vector.
    """
    #delete the corresponding bit in the vector
    if delete:
        value = 0
    else:
        value = 1
    agent = get_agent_from_hostname(hostname)
    host = get_host_from_hostname(hostname)
    index = HOSTS.index(host)
    if agent != "blue_agent_4":
        vec[index] = value
    elif hostname.startswith("admin"):
        index = HOSTS.index(host)
        vec[index] = value
    elif hostname.startswith("office"):
        index = HOSTS.index(host) + len(HOSTS)
        vec[index] = value
    elif hostname.startswith("public"):
        index = HOSTS.index(host) + len(HOSTS) *2
        vec[index] = value
    else:
        raise AssertionError(f"Hostname {hostname} or server not recognized in {agent}")
    return vec

####### OLD CODE FOR REFERENCE, NOT USED ANYMORE #######

"""'''
        Translates the action from the agent's output format to the environment's expected format.
        Args:
            name: Agent name string.
            action: Encoded action (tensor, tuple, or array).
        Returns:
            Environment action object (e.g., Monitor, Restore, etc.).
        '''
        # Possible target Subnets
        subnets = ["restricted_zone_a_subnet",
                    "operational_zone_a_subnet",
                    "restricted_zone_b_subnet",
                    "operational_zone_b_subnet",
                    "public_access_zone_subnet",
                    "admin_network_subnet",
                    "office_network_subnet"]

        session = 0 # always same

        # Handle action output from FFN
        if isinstance(action, torch.Tensor):
            action = np.array(action)
            
            # Split the action into its components: action type, host, and subnet
            a = action[:7]
            h = action[7:17+7]
            s = action[17+7:]

            a_idx = np.where(a == 1)[0]
            h_idx = np.where(h == 1)[0]
            s_idx = np.where(s == 1)[0]

            def extract_single_index(arr):
                if len(arr) == 0:
                    return None
                elif len(arr) == 1:
                    return arr[0]
                else:
                    return arr[0]  # Return the first one for now, but this should be handled more robustly
                
            a_idx = extract_single_index(a_idx)
            h_idx = extract_single_index(h_idx)
            s_idx = extract_single_index(s_idx)

        elif isinstance(action, tuple):
            a_idx = np.array(action[0])[0]
            h_idx = np.array(action[1])[0]
            s_idx = np.array(action[2])[0]
        else:
            raise ValueError

        agent_id = int(name.split("_")[-1])
        own_subnet = subnets[agent_id]
        target_subnet = subnets[s_idx] if s_idx is not None else own_subnet
        host = HOSTS[h_idx] if h_idx is not None else "router"
        hostname = target_subnet + "_" + host 

        if a_idx == 0:
            return Monitor(session=session, agent=name)
        elif a_idx == 1:
            return Restore(session=session, agent=name, hostname=hostname)
        elif a_idx == 2:
            return Remove(session=session, agent=name, hostname=hostname)
        elif a_idx == 3:
            return AllowTrafficZone(session=session, agent=name, from_subnet=target_subnet, to_subnet=own_subnet)
        elif a_idx == 4:
            return BlockTrafficZone(session=session, agent=name, from_subnet=target_subnet, to_subnet=own_subnet)
        elif a_idx == 5:
            return DeployDecoy(session=session, agent=name, hostname=hostname)
        elif a_idx == 6:
            return Analyse(session=session, agent=name, hostname=hostname)"""