from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from config import GOOGLE_DRIVE_ID

gauth = GoogleAuth(settings_file='settings.yml')
drive = GoogleDrive(gauth)

gfile = drive.CreateFile({'parents': [{'id': GOOGLE_DRIVE_ID}]})
# Read file and set it as the content of this instance.

upload_file = 'seg-data/tmp/cl1es51zc006f10981kvr6b2w.png'
gfile.SetContentFile(upload_file)
gfile.Upload()  # Upload the file.
link = gfile['webContentLink']
file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(GOOGLE_DRIVE_ID)}).GetList()
for file in file_list:
    file.Delete()
print('a')
