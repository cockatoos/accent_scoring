# 05-upload-data.py
from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()
# datastore.upload(src_dir='./data/common_voice/train',
#                  target_path='common_voice_chopped/train',
#                  overwrite=True)
# datastore.upload(src_dir='./data/common_voice/test',
#                  target_path='common_voice_chopped/test',
#                  overwrite=True)
# datastore.upload(src_dir='./data/common_voice/val',
#                  target_path='common_voice_choppedup/val',
#                  overwrite=True)
# datastore.upload(src_dir='./data/recordings/recordings',
#                  target_path='stella_raw',
#                  overwrite=True)
datastore.upload(src_dir='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-train.csv',
                 target_path='common_voice_chopped/',
                 overwrite=True)
datastore.upload(src_dir='/Users/yejinseo/Desktop/azure_ai_hack/accent_scoring/data/archive/cv-valid-test.csv',
                 target_path='common_voice_chopped/',
                 overwrite=True)
