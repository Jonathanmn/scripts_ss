import paramiko
import os
from stat import S_ISDIR

def download_roua_process(hostname, username, password, remote_path, local_path):
    """
    descarga del server process una carpeta a una ruta local
    """
    try:

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())


        ssh_client.connect(hostname, username=username, password=password)
        sftp_client = ssh_client.open_sftp()

        def _download_recursive(remote_dir, local_dir):

            os.makedirs(local_dir, exist_ok=True)
            for item in sftp_client.listdir_attr(remote_dir):
                remote_path = os.path.join(remote_dir, item.filename)
                local_path = os.path.join(local_dir, item.filename)
                if S_ISDIR(item.st_mode):
                    _download_recursive(remote_path, local_path)
                else:
                    sftp_client.get(remote_path, local_path)

        # descarga de archivos.
        _download_recursive(remote_path, local_path)


        sftp_client.close()
        ssh_client.close()

        print(f"Folder '{remote_path}' se descargo a '{local_path}'")

    except Exception as e:
        print(f"No se pudo descargar: {e}")

# Example usage (same as before)
hostname = "132.248.8.32"
username = "gei"
password = "ru04g31"
remote_path = "/home/gei/scripts_j/raw"
local_path = "/home/jmn/server_gei"

download_roua_process(hostname, username, password, remote_path, local_path)