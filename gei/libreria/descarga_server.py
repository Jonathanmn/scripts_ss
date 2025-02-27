import paramiko
import os
from stat import S_ISDIR

hostname = "132.248.8.32"
username = "gei"
password = "ru04g31"
remote_path = "/home/gei/scripts_j/l0"
local_path = "/home/jmn/l0"




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

def upload_file_to_server(hostname, username, password, local_file_path, remote_file_path):
    """
    Sube un archivo desde una ruta local a una ruta remota en el servidor
    """
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, username=username, password=password)
        sftp_client = ssh_client.open_sftp()

        # Subir el archivo
        sftp_client.put(local_file_path, remote_file_path)

        sftp_client.close()
        ssh_client.close()

        print(f"File '{local_file_path}' se subio a '{remote_file_path}'")

    except Exception as e:
        print(f"No se pudo subir el archivo: {e}")

# Argumentos para la descarga


# Descargar carpeta
download_roua_process(hostname, username, password, remote_path, local_path)

# Argumentos para la subida


local_file_path = "/home/jmn/ss/scripts/git/scripts_ss/gei/libreria/gei_raw_to_l0.py"
remote_file_path = "/home/gei/scripts_j/gei_raw_to_l0.py"

# Subir archivo
#(hostname, username, password, local_file_path, remote_file_path)



local_file_path = "/home/jmn/ss/scripts/git/scripts_ss/gei/libreria/picarro_l0_server.py"
remote_file_path = "/home/gei/scripts_j/picarro_l0_server.py"

#upload_file_to_server(hostname, username, password, local_file_path, remote_file_path)