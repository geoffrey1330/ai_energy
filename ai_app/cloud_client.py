import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


connection_str = "DefaultEndpointsProtocol=https;AccountName=csb1003bffda17e7a31;AccountKey=kY6rlrOGJXCtlj+jFFKbDqiGzif4zVcgPrKZLeNCN5XfDyKW6dx1KjswsnYBkBns6jFHdT8uY737FAqgjuGYCg==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_str)


def create_container(machine_type, section_idx):
    # Create a container for a machine
    container_name = f"{machine_type}-section-0{section_idx}"
    container_client = blob_service_client.create_container(container_name)

    return container_client.url


def upload_file(path, container_name):
    try:
        upload_file_path, local_file_name = os.path.split(path)

        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=local_file_name)

        # Upload the file
        with open(path, "rb") as data:
            blob_client.upload_blob(data)
    except Exception as e:
        print(e)


def get_most_recent_filename(container_name):
    container = blob_service_client.get_container_client(container_name)
    details = {blob.last_modified: blob.name for blob in container.list_blobs()}
    return details[sorted(details.keys())[-1]]


def download_file(container_name):
    file_name = get_most_recent_filename(container_name)
    blob_client = blob_service_client.get_blob_client(container=container_name,
                                                      blob=file_name)
    with open(file_name, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    return file_name
