import paramiko
import os
import time
import pickle

from braket.circuits import Circuit

class AwsDevice():
    """
    Amazon Braket implementation of a device using a remote server.
    """
    
    def __init__(self, hostname: str, port: int, username: str, key_path: str):
        """
        Initializes the AwsDevice with SSH connection details and remote paths.

        """
        self.hostname = hostname
        self.port = port
        self.username = username
        self.key_path = key_path
        self.remote_circuit_path = "/tmp/circuit.pkl"
        self.remote_script_path = "~/run.py"
        self.remote_result_path = "/tmp/result.pkl"

    def run(self, circuit: Circuit, shots: int = 1000):
        """
        Executes the quantum circuit on the remote server.

        """

        local_circuit_file = "/tmp/circuit.pkl"
        with open(local_circuit_file, 'wb') as f:
            pickle.dump(circuit, f, protocol=pickle.HIGHEST_PROTOCOL)

        local_result_file = "/tmp/result.pkl"

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            print(f"Connecting to {self.hostname}:{self.port} as {self.username}...")
            ssh.connect(hostname=self.hostname, port=self.port, username=self.username, key_filename=self.key_path)
            print("SSH connection established.")
        except Exception as e:
            print(f"Failed to connect to {self.hostname}:{self.port}. Error: {e}")
            os.remove(local_circuit_file)
            raise

        try:
            sftp = ssh.open_sftp()
            
            print(f"Uploading {local_circuit_file} to {self.remote_circuit_path}...")
            sftp.put(local_circuit_file, self.remote_circuit_path)
            print("Upload completed.")

            sftp.close()

            command = f"bash -c 'source ~/miniconda3/bin/activate snuqs && python3 {self.remote_script_path} --input {self.remote_circuit_path} --output {self.remote_result_path} --shots {shots}'"

            transport = ssh.get_transport()
            channel = transport.open_session()
            channel.get_pty()
            channel.exec_command(command)
            print("Remote script execution started.")

            try:
                while not channel.exit_status_ready():
                    time.sleep(0.1)
                exit_status = channel.recv_exit_status()
                if exit_status == 0:
                    print("\nRemote script executed successfully.")
                else:
                    print(f"\nRemote script failed with exit status {exit_status}.")
                    raise Exception(f"Remote script failed with exit status {exit_status}.")
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected. Attempting to cancel the remote job...")
                channel.send_signal('SIGINT')
                time.sleep(2)
                if not channel.exit_status_ready():
                    print("Remote process did not terminate gracefully. Sending SIGKILL...")
                    channel.send_signal('SIGKILL')
                print("Remote job has been cancelled.")
                raise

            print(f"Downloading {self.remote_result_path} to {local_result_file}...")
            sftp = ssh.open_sftp()
            sftp.get(self.remote_result_path, local_result_file)
            print("Download completed.")
            sftp.close()

            with open(local_result_file, 'rb') as f:
                result = pickle.load(f)
            print("Result loaded from the downloaded JSON file.")
            return result

        except Exception as e:
            print(f"An error occurred during remote operations: {e}")
            raise

        finally:
            if os.path.exists(local_circuit_file):
                os.remove(local_circuit_file)
                print(f"Local file {local_circuit_file} removed.")
            if os.path.exists(local_result_file):
                os.remove(local_result_file)
                print(f"Local file {local_result_file} removed.")
            ssh.close()
            print("SSH connection closed.")
