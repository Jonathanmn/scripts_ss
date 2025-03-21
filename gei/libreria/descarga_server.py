import paramiko
import os
from stat import S_ISDIR

hostname = "132.248.8.32"
username = "gei"
password = "ru04g31"
remote_path = "/home/gei/scripts_sandra"
local_path = "/home/jmn/server_gei/low/sandra"




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
def upload_file_or_folder_to_server(hostname, username, password, local_path, remote_path):
    """
    Sube un archivo o carpeta desde una ruta local a una ruta remota en el servidor
    """
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, username=username, password=password)
        sftp_client = ssh_client.open_sftp()

        def _upload_recursive(local_dir, remote_dir):
            os.makedirs(local_dir, exist_ok=True)
            try:
                sftp_client.mkdir(remote_dir)
            except IOError:
                pass  # Directorio ya existe

            for item in os.listdir(local_dir):
                local_path = os.path.join(local_dir, item)
                remote_path = os.path.join(remote_dir, item)
                if os.path.isdir(local_path):
                    _upload_recursive(local_path, remote_path)
                else:
                    sftp_client.put(local_path, remote_path)

        if os.path.isdir(local_path):
            _upload_recursive(local_path, remote_path)
        else:
            sftp_client.put(local_path, remote_path)

        sftp_client.close()
        ssh_client.close()

        print(f"'{local_path}' se subió a '{remote_path}'")

    except Exception as e:
        print(f"No se pudo subir '{local_path}': {e}")


# Descargar carpeta
download_roua_process(hostname, username, password, remote_path, local_path)




# Argumentos para la subida

#local_file_path = "/home/jmn/ss/scripts/git/scripts_ss/gei/libreria/gei_raw_to_l0.py"
#remote_file_path = "/home/gei/scripts_j/gei_raw_to_l0.py"

# Subir archivo
#(hostname, username, password, local_file_path, remote_file_path)



#local_file_path = "/home/jmn/ss/scripts/git/scripts_ss/gei/libreria/picarro_l0_server.py"
#remote_file_path = "/home/gei/scripts_j/picarro_l0_server.py"

#upload_file_to_server(hostname, username, password, local_file_path, remote_file_path)
#local_folder_path='/home/jmn/l0-1'
#remote_folder_path = "/home/gei/scripts_j/L0"


#upload_file_or_folder_to_server(hostname, username, password, local_folder_path, remote_folder_path)